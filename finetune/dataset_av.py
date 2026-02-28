import copy
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_MARKUP = "<image>./</image>"
AUDIO_MARKUP = "<audio>./</audio>"
PATH_REWRITES = (
    ("/mnt/lustre/work", "/weka"),
    ("/home/kuehne", "/weka/kuehne"),
)


def _resolve_media_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    for old_prefix, new_prefix in PATH_REWRITES:
        if path_str.startswith(old_prefix):
            candidate = Path(path_str.replace(old_prefix, new_prefix, 1))
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Unable to resolve media path: {path_str}")


def _prepend_to_first_user(conversations: List[Dict[str, str]], prefix: str) -> None:
    for message in conversations:
        if message["role"] == "user":
            content = message.get("content", "")
            if content:
                message["content"] = f"{prefix}\n{content}"
            else:
                message["content"] = prefix
            return
    raise ValueError("No user message found for multimodal prefix insertion.")


def _render_chat_text(tokenizer, conversations: List[Dict[str, str]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    try:
        return tokenizer.apply_chat_template(
            conversations,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(conversations, **kwargs)


def build_labels_from_input_ids(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    start_id = tokenizer.bos_token_id
    end_id = tokenizer.eos_token_id
    empty_think_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

    context = torch.ones_like(input_ids, dtype=torch.int8)
    start_positions = set(torch.where(input_ids == start_id)[0].tolist())
    assistant_positions = torch.where(input_ids == assistant_id)[0].tolist()
    end_positions = torch.where(input_ids == end_id)[0]

    for assistant_pos in assistant_positions:
        if assistant_pos - 1 not in start_positions:
            continue

        content_start = assistant_pos + 2
        if empty_think_ids:
            window = input_ids[content_start : content_start + len(empty_think_ids)].tolist()
            if window == empty_think_ids:
                content_start += len(empty_think_ids)
        end_candidates = end_positions[end_positions > content_start]
        if len(end_candidates) == 0:
            continue

        content_end = int(end_candidates[0].item())
        context[content_start : content_end + 1] = 0

    if torch.all(context):
        logger.error("No tokens available to compute loss.")
        raise ValueError("No assistant tokens available to compute loss.")

    labels = torch.full_like(input_ids, -100, dtype=torch.int32)
    eos_id = tokenizer.eos_token_id
    for idx in range(1, len(input_ids)):
        if context[idx] == 0:
            labels[idx - 1] = input_ids[idx]
        if context[idx] == 1 and context[idx - 1] == 0:
            labels[idx - 1] = eos_id

    return labels


def _pad_audio_features(examples: List[Dict[str, torch.Tensor]]) -> Union[List, torch.Tensor]:
    tensors = []
    max_frames = 0
    for example in examples:
        audio_features = example["audio_features"]
        if isinstance(audio_features, torch.Tensor) and audio_features.numel() > 0:
            tensors.append(audio_features)
            max_frames = max(max_frames, audio_features.shape[-1])

    if not tensors:
        return []

    padded = []
    for tensor in tensors:
        if tensor.shape[-1] < max_frames:
            pad = tensor.new_zeros((tensor.shape[0], tensor.shape[1], max_frames - tensor.shape[-1]))
            tensor = torch.cat([tensor, pad], dim=-1)
        padded.append(tensor)

    return torch.cat(padded, dim=0)


def av_data_collator(examples, padding_value=0, max_length=4096):
    def trim_and_pad(sequences, pad_value):
        return pad_sequence(
            [sequence[:max_length] for sequence in sequences],
            batch_first=True,
            padding_value=pad_value,
        )

    input_ids = trim_and_pad([example["input_ids"] for example in examples], padding_value)
    position_ids = trim_and_pad([example["position_ids"] for example in examples], padding_value)
    labels = trim_and_pad([example["labels"] for example in examples], -100)
    attention_mask = trim_and_pad([example["attention_mask"] for example in examples], 0).bool()

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": [example["pixel_values"] for example in examples],
        "tgt_sizes": [example["tgt_sizes"] for example in examples],
        "image_bound": [example["image_bound"] for example in examples],
        "audio_features": _pad_audio_features(examples),
        "audio_feature_lens": [example["audio_feature_lens"] for example in examples],
        "audio_bounds": [example["audio_bounds"] for example in examples],
        "spk_bounds": [example["spk_bounds"] for example in examples],
    }


class AVSupervisedDataset(Dataset):
    """Supervised fine-tuning dataset for audio-visual to text tasks."""

    def __init__(
        self,
        raw_data,
        processor,
        tokenizer,
        max_length=4096,
        max_slice_nums=1,
    ):
        super().__init__()
        self.raw_data = raw_data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_slice_nums = max_slice_nums

    def __len__(self):
        return len(self.raw_data)

    def _prepare_single_image(
        self,
        image_path: str,
        conversations: List[Dict[str, str]],
    ) -> List[Image.Image]:
        image = Image.open(_resolve_media_path(image_path)).convert("RGB")
        image_count = 0

        for message in conversations:
            content = message.get("content", "")
            image_count += content.count("<image>")
            message["content"] = content.replace("<image>", IMAGE_MARKUP)

        if image_count == 0:
            image_count = 1
            _prepend_to_first_user(conversations, IMAGE_MARKUP)

        return [image.copy() for _ in range(image_count)]

    def _prepare_multi_image(
        self,
        image_map: Dict[str, str],
        conversations: List[Dict[str, str]],
    ) -> List[Image.Image]:
        loaded_images = {
            placeholder: Image.open(_resolve_media_path(path)).convert("RGB")
            for placeholder, path in image_map.items()
        }
        ordered_images = []
        placeholder_pattern = re.compile(r"<image_\d+>")

        for message in conversations:
            content = message.get("content", "")

            def replace(match):
                placeholder = match.group(0)
                if placeholder not in loaded_images:
                    raise ValueError(f"Unknown image placeholder: {placeholder}")
                ordered_images.append(loaded_images[placeholder].copy())
                return IMAGE_MARKUP

            message["content"] = placeholder_pattern.sub(replace, content)

        if not ordered_images:
            ordered_keys = sorted(loaded_images)
            prefix = "\n".join(IMAGE_MARKUP for _ in ordered_keys)
            _prepend_to_first_user(conversations, prefix)
            ordered_images = [loaded_images[key].copy() for key in ordered_keys]

        return ordered_images

    def _prepare_images(
        self,
        image_spec: Optional[Union[str, Dict[str, str]]],
        conversations: List[Dict[str, str]],
    ) -> Optional[List[Image.Image]]:
        if image_spec is None:
            return None
        if isinstance(image_spec, str):
            return self._prepare_single_image(image_spec, conversations)
        if isinstance(image_spec, dict):
            return self._prepare_multi_image(image_spec, conversations)
        raise TypeError(f"Unsupported image spec type: {type(image_spec)!r}")

    def _prepare_audio(
        self,
        audio_spec: Optional[str],
        conversations: List[Dict[str, str]],
    ) -> Optional[List]:
        if audio_spec is None:
            return None
        if not isinstance(audio_spec, str):
            raise TypeError(f"Unsupported audio spec type: {type(audio_spec)!r}")

        audio, _ = librosa.load(_resolve_media_path(audio_spec), sr=16000, mono=True)
        audio_count = 0
        for message in conversations:
            content = message.get("content", "")
            audio_count += content.count("<audio>")
            message["content"] = content.replace("<audio>", AUDIO_MARKUP)

        if audio_count == 0:
            audio_count = 1
            _prepend_to_first_user(conversations, AUDIO_MARKUP)

        return [audio.copy() for _ in range(audio_count)]

    def _prepare_sample(self, sample: Dict) -> Tuple[List[Dict[str, str]], Optional[List[Image.Image]], Optional[List]]:
        conversations = copy.deepcopy(sample["conversations"])
        if len(conversations) < 2:
            raise ValueError("conversations length must be at least 2")
        if conversations[0]["role"] != "user":
            raise ValueError("the first role must be user")

        images = self._prepare_images(sample.get("image"), conversations)
        audios = self._prepare_audio(sample.get("audio"), conversations)
        return conversations, images, audios

    def __getitem__(self, index):
        try:
            sample = self.raw_data[index]
            conversations, images, audios = self._prepare_sample(sample)
            text = _render_chat_text(self.tokenizer, conversations)
            batch = self.processor(
                text=text,
                images=images,
                audios=audios,
                max_length=self.max_length,
                max_slice_nums=self.max_slice_nums,
                return_tensors="pt",
            )

            input_ids = batch["input_ids"][0].to(dtype=torch.int32)
            labels = build_labels_from_input_ids(input_ids, self.tokenizer)
            position_ids = torch.arange(input_ids.size(0), dtype=torch.long)

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
                "pixel_values": batch["pixel_values"][0] if batch["pixel_values"] else [],
                "tgt_sizes": batch["tgt_sizes"][0] if batch["tgt_sizes"] else [],
                "image_bound": batch["image_bound"][0] if batch["image_bound"] else [],
                "audio_features": batch.get("audio_features", []),
                "audio_feature_lens": batch["audio_feature_lens"][0] if batch.get("audio_feature_lens") else [],
                "audio_bounds": batch["audio_bounds"][0] if batch["audio_bounds"] else [],
                "spk_bounds": batch["spk_bounds"][0] if batch["spk_bounds"] else [],
            }
        except Exception:
            logger.exception("data fetch error")
            if len(self.raw_data) == 1:
                raise
            return self.__getitem__(random.randint(0, len(self.raw_data) - 1))
