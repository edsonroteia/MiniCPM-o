#!/usr/bin/env python3
"""Preprocess SFT training data to replace letter answers with choice text.

For ~50% of samples, replaces `<answer>C</answer>` with `<answer>sound of wind</answer>`
by parsing the multi-choice options from the user prompt.

Usage:
    python preprocess_text_answers.py \
        --input sample_data/av_reasoning_full_video/train.json \
        --output sample_data/av_reasoning_full_video/train_textanswer.json \
        --fraction 0.5 \
        --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys


OPTION_PATTERN = re.compile(r"([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)", re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def parse_options(user_content: str) -> dict[str, str]:
    """Extract {letter: text} from user prompt."""
    matches = OPTION_PATTERN.findall(user_content)
    return {letter.upper(): text.strip() for letter, text in matches}


def replace_answer_letter(assistant_content: str, options: dict[str, str]) -> str | None:
    """Replace <answer>LETTER</answer> with <answer>choice text</answer>.

    Returns the modified string, or None if no replacement was possible.
    """
    match = ANSWER_TAG_PATTERN.search(assistant_content)
    if not match:
        return None

    letter = match.group(1).upper()
    choice_text = options.get(letter)
    if not choice_text:
        return None

    return assistant_content[: match.start()] + f"<answer>{choice_text}</answer>" + assistant_content[match.end() :]


def main():
    parser = argparse.ArgumentParser(description="Replace letter answers with choice text in SFT data.")
    parser.add_argument("--input", required=True, help="Input train.json path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction of samples to convert (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    rng = random.Random(args.seed)
    converted = 0
    skipped = 0
    total = len(data)

    output_data = []
    for sample in data:
        sample = copy.deepcopy(sample)
        conversations = sample.get("conversations", [])

        if len(conversations) < 2 or rng.random() > args.fraction:
            output_data.append(sample)
            continue

        user_content = conversations[0].get("content", "")
        asst_content = conversations[1].get("content", "")
        options = parse_options(user_content)

        if not options:
            output_data.append(sample)
            skipped += 1
            continue

        new_asst = replace_answer_letter(asst_content, options)
        if new_asst is None:
            output_data.append(sample)
            skipped += 1
            continue

        conversations[1]["content"] = new_asst
        output_data.append(sample)
        converted += 1

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Total: {total}, Converted: {converted}, Skipped: {skipped}, Unchanged: {total - converted - skipped}")
    # Show a few examples
    for sample in output_data[:3]:
        asst = sample["conversations"][1]["content"]
        match = ANSWER_TAG_PATTERN.search(asst)
        tag_match = re.search(r"<answer>(.*?)</answer>", asst, re.DOTALL | re.IGNORECASE)
        if tag_match:
            print(f"  id={sample['id']}: <answer>{tag_match.group(1)}</answer>")


if __name__ == "__main__":
    main()
