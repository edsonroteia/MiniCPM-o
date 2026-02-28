import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset


AVQA_ROOT = Path("/weka/kuehne/kqr867/datasets/AVQA")
LETTERS = ["A", "B", "C", "D"]


def _resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    if path_str.startswith("/mnt/lustre/work"):
        candidate = Path(path_str.replace("/mnt/lustre/work", "/weka", 1))
        if candidate.exists():
            return candidate

    if path_str.startswith("/home/kuehne"):
        candidate = Path(path_str.replace("/home/kuehne", "/weka/kuehne", 1))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Unable to resolve path: {path_str}")


def _resolve_video_path(record: Dict[str, Any]) -> Path:
    audio_path = _resolve_existing_path(record["audio_path"])
    split = "eval" if "/eval/" in audio_path.as_posix() else "train"
    video_path = AVQA_ROOT / split / f"{record['video_name']}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing AVQA video: {video_path}")
    return video_path


def _build_image_placeholders(num_video_frames: int) -> List[str]:
    return [f"<image_{idx:02d}>" for idx in range(num_video_frames)]


def _format_choices(choices: List[Any]) -> List[str]:
    return [f"{LETTERS[idx]}. {choice}" for idx, choice in enumerate(choices[:4])]


def build_user_prompt_text(record: Dict[str, Any], num_video_frames: int) -> str:
    lines = _build_image_placeholders(num_video_frames)
    lines.append("<audio>")
    lines.append(record["question_text"])
    lines.extend(_format_choices(record["multi_choice"]))
    return "\n".join(lines)


def build_prompt_messages(record: Dict[str, Any], num_video_frames: int) -> List[Dict[str, str]]:
    return [{"role": "user", "content": build_user_prompt_text(record, num_video_frames)}]


def build_avqa_grpo_dataset(
    data_path: str,
    num_video_frames: int = 8,
    max_samples: Optional[int] = None,
) -> Dataset:
    rows = []
    with open(data_path, "r", encoding="utf8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            audio_path = _resolve_existing_path(record["audio_path"]).resolve()
            video_path = _resolve_video_path(record).resolve()
            answer_idx = int(record["answer"])
            answer_letter = LETTERS[answer_idx]

            rows.append(
                {
                    "sample_id": str(record.get("id", idx)),
                    "prompt": build_prompt_messages(record, num_video_frames),
                    "solution": f"<answer>{answer_letter}</answer>",
                    "answer_letter": answer_letter,
                    "multi_choice": list(record["multi_choice"]),
                    "question_text": record["question_text"],
                    "video": str(video_path),
                    "audio": str(audio_path),
                    "audio_path": str(audio_path),
                    "video_path": str(video_path),
                }
            )

            if max_samples is not None and max_samples > 0 and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError("No AVQA rows were loaded for GRPO.")

    return Dataset.from_list(rows)
