import math
import re
from typing import Any, Callable, Dict, List, Tuple


LETTERS = ["A", "B", "C", "D"]


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return str(item.get("content", ""))
        return str(item)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _extract_letter_answer(candidate: str) -> str | None:
    candidate = candidate.strip()
    match = re.match(r"^\(?\s*([A-D])\s*\)?[\.\!\?,;:]*\s*$", candidate, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([A-D])\s*[\.\)\-:]\s*", candidate, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])(?:[\.\!\?,;:]*)\s*$", candidate, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _extract_answer_tag(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        tagged = match.group(1).strip()
        tagged_letter = _extract_letter_answer(tagged)
        return tagged_letter if tagged_letter is not None else tagged
    stripped = text.strip()
    direct = _extract_letter_answer(stripped)
    if direct is not None:
        return direct
    return stripped


def _normalize_text_for_match(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _map_candidate_to_letter(candidate: str, choice_list: list[Any] | None) -> str | None:
    cleaned = candidate.strip().strip(" \t\r\n\"'`[](){}.,!?;:")
    if not cleaned:
        return None

    direct_letter = _extract_letter_answer(cleaned)
    if direct_letter is not None:
        return direct_letter

    if not choice_list:
        return None

    normalized_candidate = _normalize_text_for_match(cleaned)
    if not normalized_candidate:
        return None

    matches = []
    for idx, choice in enumerate(choice_list[:4]):
        normalized_choice = _normalize_text_for_match(str(choice))
        if normalized_candidate == normalized_choice:
            matches.append(idx)

    if len(matches) == 1:
        return LETTERS[matches[0]]
    return None


def _extract_committed_choice(text: str, choice_list: list[Any] | None) -> str | None:
    tagged_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if tagged_match:
        letter = _map_candidate_to_letter(tagged_match.group(1), choice_list)
        if letter is not None:
            return letter

    stripped = text.strip()

    if len(stripped.split()) <= 4:
        letter = _map_candidate_to_letter(stripped, choice_list)
        if letter is not None:
            return letter

    tail_text = stripped[-200:]
    cue_patterns = [
        r"(?:final answer|correct answer|correct option|the answer|answer|option|choice)\s*(?:is|:)\s*(?P<candidate>[^\n<]{1,80})",
        r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer|correct answer|correct option)?\s*(?:is|:)\s*(?P<candidate>[^\n<]{1,80})",
    ]
    for pattern in cue_patterns:
        matches = list(re.finditer(pattern, tail_text, re.IGNORECASE))
        if not matches:
            continue
        candidate = matches[-1].group("candidate")
        candidate = re.split(r"[\n]", candidate, maxsplit=1)[0]
        letter = _map_candidate_to_letter(candidate, choice_list)
        if letter is not None:
            return letter

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line.split()) <= 6:
            letter = _map_candidate_to_letter(last_line, choice_list)
            if letter is not None:
                return letter

    return None


def _broadcast_choice_arg(choice_arg: Any, batch_size: int) -> list[Any]:
    if choice_arg is None:
        return [None] * batch_size

    if batch_size == 1 and isinstance(choice_arg, list):
        if not choice_arg or not isinstance(choice_arg[0], list):
            return [choice_arg]

    if isinstance(choice_arg, list) and len(choice_arg) == batch_size:
        return choice_arg

    return [None] * batch_size


def accuracy_reward(completions, solution, multi_choice=None, choices=None, **kwargs):
    contents = [_extract_completion_text(completion) for completion in completions]
    rewards = []
    multi_choice_list = _broadcast_choice_arg(multi_choice, len(contents))
    choices_list = _broadcast_choice_arg(choices, len(contents))

    for content, sol, mc, choice_alias in zip(contents, solution, multi_choice_list, choices_list):
        choice_list = mc if mc is not None else choice_alias
        expected_choice = _extract_committed_choice(str(sol), choice_list)
        predicted_choice = _extract_committed_choice(content, choice_list)
        if expected_choice is not None and predicted_choice == expected_choice:
            rewards.append(1.0)
            continue

        rewards.append(1.0 if _extract_answer_tag(content) == _extract_answer_tag(str(sol)) else 0.0)

    return rewards


def format_reward(completions, **kwargs):
    completion_contents = [_extract_completion_text(completion) for completion in completions]
    rewards = []

    strict_pattern = re.compile(
        r"\s*<think>.*?</think>\s*<answer>\s*(.+?)\s*</answer>\s*\Z",
        re.DOTALL | re.IGNORECASE,
    )
    answer_span_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    for content in completion_contents:
        if strict_pattern.fullmatch(content):
            rewards.append(1.0)
            continue

        reward = 0.0
        think_close_spans = list(re.finditer(r"</think>", content, re.IGNORECASE))
        answer_open_spans = list(re.finditer(r"<answer>", content, re.IGNORECASE))
        answer_close_spans = list(re.finditer(r"</answer>", content, re.IGNORECASE))
        answer_spans = list(answer_span_pattern.finditer(content))

        if len(think_close_spans) == 1:
            reward += 0.25
        elif think_close_spans:
            reward += 0.10

        if len(answer_open_spans) == 1:
            reward += 0.20
        elif answer_open_spans:
            reward += 0.05

        if len(answer_close_spans) == 1:
            reward += 0.20
        elif answer_close_spans:
            reward += 0.05

        if think_close_spans and answer_open_spans and think_close_spans[0].end() <= answer_open_spans[0].start():
            reward += 0.20

        if len(answer_spans) == 1:
            answer_text = answer_spans[0].group(1).strip()
            reward += 0.15 if answer_text else 0.05
        elif len(answer_spans) > 1:
            if any(_extract_letter_answer(match.group(1).strip()) is not None for match in answer_spans):
                reward += 0.05

        rewards.append(min(reward, 1.0))

    return rewards


def reasoning_length_reward(completions, **kwargs):
    contents = [_extract_completion_text(completion) for completion in completions]
    rewards = []

    for content in contents:
        reward = 0.0
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_text = think_match.group(1).strip()
            word_count = len(thinking_text.split())
            optimal_length = 75
            sigma = 15
            gaussian_reward = math.exp(-((word_count - optimal_length) ** 2) / (2 * sigma**2))
            range_bonus = 0.2 if 50 <= word_count <= 100 else 0.0
            reward = min(gaussian_reward + range_bonus, 1.0)
        rewards.append(reward)

    return rewards


REWARD_REGISTRY: Dict[str, Callable[..., List[float]]] = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_length": reasoning_length_reward,
}


def resolve_reward_functions(names: str) -> Tuple[List[Callable[..., List[float]]], List[float], List[str]]:
    reward_names = [name.strip() for name in names.split() if name.strip()]
    if not reward_names:
        raise ValueError("At least one reward function must be specified.")

    reward_funcs = []
    for name in reward_names:
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward function '{name}'. Available: {sorted(REWARD_REGISTRY)}")
        reward_funcs.append(REWARD_REGISTRY[name])

    if "accuracy" in reward_names:
        num_other_rewards = len(reward_names) - 1
        other_weight = 0.2 / num_other_rewards if num_other_rewards > 0 else 0.0
        reward_weights = [0.8 if name == "accuracy" else other_weight for name in reward_names]
    else:
        equal_weight = 1.0 / len(reward_names)
        reward_weights = [equal_weight] * len(reward_names)

    return reward_funcs, reward_weights, reward_names
