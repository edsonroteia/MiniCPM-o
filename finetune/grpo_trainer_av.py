import copy
import logging
from collections import deque
from typing import Any, Dict, List, Optional

import librosa
import torch
from PIL import Image
import trl.trainer.grpo_trainer as trl_grpo_module
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import (
    FSDP,
    disable_gradient_checkpointing,
    entropy_from_logits,
    gather_object,
    nanmax,
    nanmin,
    nanstd,
    nullcontext,
    profiling_decorator,
    selective_log_softmax,
    unwrap_model_for_generation,
    use_adapter,
)
from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

from dataset_av import (
    AUDIO_MARKUP,
    IMAGE_MARKUP,
    IMAGE_PLACEHOLDER_RE,
    _pad_audio_features,
    _prepend_to_first_user,
    _resolve_media_path,
)


logger = logging.getLogger(__name__)


def render_grpo_chat_text(tokenizer, conversations: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(conversations, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(conversations, **kwargs)


def _build_raw_prompt_text(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        if message.get("role") == "user":
            parts.append(message.get("content", ""))
    return "\n\n".join(part for part in parts if part)


def _load_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError(
            "decord is required to load video samples for GRPO. Use the minicpm environment."
        ) from exc

    video_reader = VideoReader(str(_resolve_media_path(video_path)), ctx=cpu(0))
    total_frames = len(video_reader)
    if total_frames < 1:
        raise ValueError(f"Video contains no frames: {video_path}")

    if total_frames == 1:
        frame_indices = [0] * num_frames
    else:
        frame_indices = [
            int(round((idx + 1) * (total_frames - 1) / (num_frames + 1)))
            for idx in range(num_frames)
        ]

    frames = video_reader.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def _prepare_images(video_path: str, conversations: List[Dict[str, str]], num_video_frames: int) -> List[Image.Image]:
    frame_count = 0
    for message in conversations:
        content = message.get("content", "")
        frame_count += len(IMAGE_PLACEHOLDER_RE.findall(content))
        message["content"] = IMAGE_PLACEHOLDER_RE.sub(IMAGE_MARKUP, content)

    if frame_count == 0:
        frame_count = num_video_frames
        prefix = "\n".join(IMAGE_MARKUP for _ in range(frame_count))
        _prepend_to_first_user(conversations, prefix)

    return _load_video_frames(video_path, frame_count)


def _prepare_audio(audio_path: str, conversations: List[Dict[str, str]]) -> List[Any]:
    audio, _ = librosa.load(_resolve_media_path(audio_path), sr=16000, mono=True)
    audio_count = 0
    for message in conversations:
        content = message.get("content", "")
        audio_count += content.count("<audio>")
        message["content"] = content.replace("<audio>", AUDIO_MARKUP)

    if audio_count == 0:
        audio_count = 1
        _prepend_to_first_user(conversations, AUDIO_MARKUP)

    return [audio.copy() for _ in range(audio_count)]


def prepare_grpo_prompt_example(
    row: Dict[str, Any],
    processor,
    tokenizer,
    max_prompt_length: Optional[int],
    max_slice_nums: int,
    num_video_frames: int,
) -> Dict[str, Any]:
    conversations = copy.deepcopy(row["prompt"])
    video_path = row.get("video") or row.get("video_path")
    audio_path = row.get("audio") or row.get("audio_path")
    if not video_path:
        raise ValueError("GRPO row is missing the video path.")
    if not audio_path:
        raise ValueError("GRPO row is missing the audio path.")

    images = _prepare_images(video_path, conversations, num_video_frames)
    audios = _prepare_audio(audio_path, conversations)
    rendered_prompt = render_grpo_chat_text(tokenizer, conversations, add_generation_prompt=True)
    processor_outputs = processor(
        text=rendered_prompt,
        images=images,
        audios=audios,
        max_length=max_prompt_length,
        max_slice_nums=max_slice_nums,
        return_tensors="pt",
    )

    prompt_input_ids = processor_outputs["input_ids"][0].to(dtype=torch.long)
    attention_mask = (
        processor_outputs["attention_mask"][0].to(dtype=torch.long)
        if "attention_mask" in processor_outputs
        else torch.ones_like(prompt_input_ids, dtype=torch.long)
    )

    return {
        "rendered_prompt": rendered_prompt,
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": attention_mask,
        "pixel_values": processor_outputs["pixel_values"][0] if processor_outputs["pixel_values"] else [],
        "tgt_sizes": processor_outputs["tgt_sizes"][0] if processor_outputs["tgt_sizes"] else [],
        "image_bound": processor_outputs["image_bound"][0] if processor_outputs["image_bound"] else [],
        "audio_features": processor_outputs.get("audio_features", []),
        "audio_feature_lens": processor_outputs["audio_feature_lens"][0]
        if processor_outputs.get("audio_feature_lens")
        else [],
        "audio_bounds": processor_outputs["audio_bounds"][0] if processor_outputs["audio_bounds"] else [],
        "spk_bounds": processor_outputs["spk_bounds"][0] if processor_outputs["spk_bounds"] else [],
    }


def _left_pad_tensors(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(sequence.size(0) for sequence in sequences)
    output = sequences[0].new_full((len(sequences), max_len), pad_value)
    for idx, sequence in enumerate(sequences):
        output[idx, -sequence.size(0) :] = sequence
    return output


def _normalize_generated_sequence(generated, reference_tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(generated, tuple):
        if len(generated) >= 2 and hasattr(generated[1], "sequences"):
            return _normalize_generated_sequence(generated[1].sequences, reference_tensor)
        if generated:
            return _normalize_generated_sequence(generated[0], reference_tensor)
        raise TypeError("Unsupported empty tuple returned by generate().")

    if torch.is_tensor(generated):
        return generated[0] if generated.dim() > 1 else generated

    if isinstance(generated, list):
        first = generated[0] if generated else []
        if torch.is_tensor(first):
            return first
        return reference_tensor.new_tensor(first, dtype=reference_tensor.dtype)

    raise TypeError(f"Unsupported generate() output type: {type(generated)!r}")


class MiniCPMOAVGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, num_video_frames: int = 8, max_slice_nums: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_video_frames = num_video_frames
        self.max_slice_nums = max_slice_nums
        self.max_prompt_length = self.args.max_prompt_length
        self.prompt_tokenizer = (
            self.processing_class.tokenizer if hasattr(self.processing_class, "tokenizer") else self.processing_class
        )
        extra_columns = [
            "audio",
            "audio_path",
            "video",
            "video_path",
            "solution",
            "answer_letter",
            "multi_choice",
            "sample_id",
            "question_text",
        ]
        self._signature_columns = list(dict.fromkeys((self._signature_columns or []) + extra_columns))
        self._trace_log_limit = max(1, int(getattr(self.args, "wandb_num_logged_completions", 8)))
        self._trace_rows = deque(maxlen=self._trace_log_limit)

    def _move_to_device(self, value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, dict):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        return value

    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._generate_and_score_completions(generation_batch)

    def _build_generation_inputs(self, prepared_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_ids = _left_pad_tensors([example["prompt_input_ids"] for example in prepared_examples], self.pad_token_id)
        prompt_mask = _left_pad_tensors([example["prompt_attention_mask"] for example in prepared_examples], 0)
        position_ids = prompt_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(prompt_mask == 0, 0)

        generation_inputs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "position_ids": position_ids,
            "pixel_values": [example["pixel_values"] for example in prepared_examples],
            "tgt_sizes": [example["tgt_sizes"] for example in prepared_examples],
            "image_bound": [example["image_bound"] for example in prepared_examples],
            "audio_features": _pad_audio_features(prepared_examples),
            "audio_feature_lens": [example["audio_feature_lens"] for example in prepared_examples],
            "audio_bounds": [example["audio_bounds"] for example in prepared_examples],
            "spk_bounds": [example["spk_bounds"] for example in prepared_examples],
        }
        return generation_inputs

    def _append_trace_logs(
        self,
        inputs: List[Dict[str, Any]],
        completions_text: List[str],
        rewards_per_func: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        if not self.accelerator.is_main_process:
            return

        reward_weights = torch.as_tensor(
            self.reward_weights,
            device=rewards_per_func.device,
            dtype=rewards_per_func.dtype,
        )
        row_count = min(len(inputs), len(completions_text), self._trace_log_limit)
        for idx in range(row_count):
            per_reward = {
                reward_name: float(rewards_per_func[idx, reward_idx].item())
                for reward_idx, reward_name in enumerate(self.reward_func_names)
            }
            weighted_reward = float(torch.nan_to_num((rewards_per_func[idx] * reward_weights).nansum(), nan=0.0).item())
            advantage = float(advantages[idx].item()) if advantages.ndim == 1 else float(advantages[idx].mean().item())

            self._trace_rows.append(
                {
                    "step": int(self.state.global_step),
                    "rank": int(self.accelerator.process_index),
                    "prompt_slot": int(idx),
                    "sample_id": str(inputs[idx].get("sample_id", "")),
                    "prompt": _build_raw_prompt_text(inputs[idx].get("prompt", [])),
                    "question": str(inputs[idx].get("question_text", "")),
                    "target": str(inputs[idx].get("solution", "")),
                    "completion": completions_text[idx],
                    "weighted_reward": weighted_reward,
                    "advantage": advantage,
                    "per_reward": per_reward,
                }
            )

    def _log_wandb_traces(self) -> None:
        report_to = self.args.report_to
        if not report_to:
            return
        if isinstance(report_to, str):
            report_targets = {report_to}
        else:
            report_targets = set(report_to)
        if "wandb" not in report_targets:
            return

        local_rows = list(self._trace_rows)
        gathered_rows = gather_object(local_rows)
        self._trace_rows.clear()

        if not self.accelerator.is_main_process or not gathered_rows:
            return

        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        columns = [
            "step",
            "prompt_slot",
            "rollout_index",
            "rank",
            "sample_id",
            "prompt",
            "question",
            "target",
            "completion",
            "weighted_reward",
            "advantage",
        ] + [f"reward/{name}" for name in self.reward_func_names]
        data = []
        gathered_rows = sorted(
            gathered_rows,
            key=lambda row: (row["step"], row["prompt_slot"], row["rank"], row["sample_id"]),
        )
        rollout_index_by_group = {}
        for trace_row in gathered_rows:
            group_key = (trace_row["step"], trace_row["prompt_slot"])
            rollout_index = rollout_index_by_group.get(group_key, 0)
            rollout_index_by_group[group_key] = rollout_index + 1
            table_row = [
                trace_row["step"],
                trace_row["prompt_slot"],
                rollout_index,
                trace_row["rank"],
                trace_row["sample_id"],
                trace_row["prompt"],
                trace_row["question"],
                trace_row["target"],
                trace_row["completion"],
                trace_row["weighted_reward"],
                trace_row["advantage"],
            ]
            for reward_name in self.reward_func_names:
                table_row.append(trace_row["per_reward"].get(reward_name))
            data.append(table_row)

        wandb.log({"grpo_traces": wandb.Table(columns=columns, data=data)})

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        original_print_sample = getattr(trl_grpo_module, "print_prompt_completions_sample", None)
        try:
            if original_print_sample is not None:
                trl_grpo_module.print_prompt_completions_sample = lambda *args, **kwargs: None
            super().log(logs, start_time)
        finally:
            if original_print_sample is not None:
                trl_grpo_module.print_prompt_completions_sample = original_print_sample
        self._log_wandb_traces()

    def _forward_minicpm(self, model, model_inputs: Dict[str, Any]):
        if hasattr(model, "_enable_peft_forward_hooks"):
            with model._enable_peft_forward_hooks(**model_inputs):
                return model.base_model(data=model_inputs, use_cache=False)
        return model(data=model_inputs, use_cache=False)

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        audio_features=None,
        audio_feature_lens=None,
        audio_bounds=None,
        spk_bounds=None,
    ):
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            position_ids_batch = attention_mask_batch.long().cumsum(-1) - 1
            position_ids_batch.masked_fill_(attention_mask_batch == 0, 0)

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "position_ids": position_ids_batch,
            }
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]
            if tgt_sizes is not None:
                model_inputs["tgt_sizes"] = tgt_sizes[start:end]
            if image_bound is not None:
                model_inputs["image_bound"] = image_bound[start:end]
            if audio_features is not None:
                if torch.is_tensor(audio_features):
                    model_inputs["audio_features"] = audio_features[start:end]
                else:
                    audio_feature_slice = audio_features[start:end]
                    model_inputs["audio_features"] = _pad_audio_features(
                        [{"audio_features": features} for features in audio_feature_slice]
                    )
            if audio_feature_lens is not None:
                model_inputs["audio_feature_lens"] = audio_feature_lens[start:end]
            if audio_bounds is not None:
                model_inputs["audio_bounds"] = audio_bounds[start:end]
            if spk_bounds is not None:
                model_inputs["spk_bounds"] = spk_bounds[start:end]

            logits = self._forward_minicpm(model, model_inputs).logits
            if logits.size(1) == input_ids_batch.size(1):
                logits = logits[:, :-1, :]
            elif logits.size(1) > input_ids_batch.size(1) - 1:
                logits = logits[:, : input_ids_batch.size(1) - 1, :]

            tokens_to_score = min(logits_to_keep, logits.size(1))
            if tokens_to_score < 1:
                raise ValueError(
                    "No completion tokens are available for GRPO log-prob scoring: "
                    f"input_ids_batch={tuple(input_ids_batch.shape)} "
                    f"logits={tuple(logits.shape)} logits_to_keep={logits_to_keep}"
                )

            logits = logits[:, -tokens_to_score:, :]
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -tokens_to_score:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    all_entropies.append(entropy_from_logits(logits))

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    @profiling_decorator
    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.use_vllm:
            raise NotImplementedError("The minimal MiniCPM-o GRPO path does not support vLLM generation.")
        if self.tools:
            raise NotImplementedError("The minimal MiniCPM-o GRPO path does not support tool-calling.")

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [example["prompt"] for example in inputs]
        prepared_examples = [
            prepare_grpo_prompt_example(
                example,
                processor=self.processing_class,
                tokenizer=self.prompt_tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_slice_nums=self.max_slice_nums,
                num_video_frames=self.num_video_frames,
            )
            for example in inputs
        ]
        rendered_prompts = [example["rendered_prompt"] for example in prepared_examples]

        generation_inputs = self._build_generation_inputs(prepared_examples)
        prompt_ids = generation_inputs["input_ids"]
        prompt_mask = generation_inputs["attention_mask"]
        generation_inputs = self._move_to_device(generation_inputs, device)

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            generated_completions = []
            extra_generate_kwargs = {}
            if isinstance(self.generation_kwargs, dict) and self.generation_kwargs.get("min_new_tokens") is not None:
                extra_generate_kwargs["min_new_tokens"] = self.generation_kwargs["min_new_tokens"]
            for idx in range(prompt_ids.size(0)):
                single_generation_inputs = {
                    "input_ids": generation_inputs["input_ids"][idx : idx + 1],
                    "attention_mask": generation_inputs["attention_mask"][idx : idx + 1],
                    "pixel_values": generation_inputs["pixel_values"][idx : idx + 1],
                    "tgt_sizes": generation_inputs["tgt_sizes"][idx : idx + 1],
                    "image_bound": generation_inputs["image_bound"][idx : idx + 1],
                    "audio_feature_lens": generation_inputs["audio_feature_lens"][idx : idx + 1],
                    "audio_bounds": generation_inputs["audio_bounds"][idx : idx + 1],
                    "spk_bounds": generation_inputs["spk_bounds"][idx : idx + 1],
                }
                if torch.is_tensor(generation_inputs["audio_features"]):
                    single_generation_inputs["audio_features"] = generation_inputs["audio_features"][idx : idx + 1]
                else:
                    single_generation_inputs["audio_features"] = generation_inputs["audio_features"]

                generated = unwrapped_model.generate(
                    **single_generation_inputs,
                    tokenizer=self.prompt_tokenizer,
                    generation_config=self.generation_config,
                    disable_compile=True,
                    **extra_generate_kwargs,
                )
                generated_completions.append(_normalize_generated_sequence(generated, generation_inputs["input_ids"]))

        max_completion_len = max((seq.size(0) for seq in generated_completions), default=0)
        completion_ids = prompt_ids.new_full((len(generated_completions), max_completion_len), self.pad_token_id).to(device)
        completion_mask = prompt_mask.new_zeros((len(generated_completions), max_completion_len)).to(device)

        for idx, sequence in enumerate(generated_completions):
            generated_completion = sequence
            if generated_completion.numel() == 0:
                continue

            generated_completion = generated_completion.to(device)
            completion_ids[idx, : generated_completion.size(0)] = generated_completion

            effective_len = generated_completion.size(0)
            eos_positions = (generated_completion == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                effective_len = int(eos_positions[0].item()) + 1
            completion_mask[idx, :effective_len] = 1

        prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]

        prompt_completion_ids = torch.cat([prompt_ids.to(device), completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask.to(device), completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    pixel_values=generation_inputs["pixel_values"],
                    tgt_sizes=generation_inputs["tgt_sizes"],
                    image_bound=generation_inputs["image_bound"],
                    audio_features=generation_inputs["audio_features"],
                    audio_feature_lens=generation_inputs["audio_feature_lens"],
                    audio_bounds=generation_inputs["audio_bounds"],
                    spk_bounds=generation_inputs["spk_bounds"],
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=generation_inputs["pixel_values"],
                        tgt_sizes=generation_inputs["tgt_sizes"],
                        image_bound=generation_inputs["image_bound"],
                        audio_features=generation_inputs["audio_features"],
                        audio_feature_lens=generation_inputs["audio_feature_lens"],
                        audio_bounds=generation_inputs["audio_bounds"],
                        spk_bounds=generation_inputs["spk_bounds"],
                    )
                else:
                    unwrapped_policy = self.accelerator.unwrap_model(self.model)
                    peft_config = getattr(unwrapped_policy, "peft_config", None)
                    if peft_config is not None:
                        with use_adapter(unwrapped_policy, adapter_name="ref" if "ref" in peft_config else None):
                            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                pixel_values=generation_inputs["pixel_values"],
                                tgt_sizes=generation_inputs["tgt_sizes"],
                                image_bound=generation_inputs["image_bound"],
                                audio_features=generation_inputs["audio_features"],
                                audio_feature_lens=generation_inputs["audio_feature_lens"],
                                audio_bounds=generation_inputs["audio_bounds"],
                                spk_bounds=generation_inputs["spk_bounds"],
                            )
                    else:
                        # Full-FT: no reference model available; skip KL
                        logger.warning(
                            "beta=%.4f but no ref_model and no peft_config found. "
                            "Setting ref_per_token_logps=None (KL will be skipped).",
                            self.beta,
                        )
                        ref_per_token_logps = None
            else:
                ref_per_token_logps = None

        scored_token_count = completion_ids.size(1)
        if old_per_token_logps is not None:
            scored_token_count = min(scored_token_count, old_per_token_logps.size(1))
        if ref_per_token_logps is not None:
            scored_token_count = min(scored_token_count, ref_per_token_logps.size(1))
        if scored_token_count < completion_ids.size(1):
            completion_ids = completion_ids[:, -scored_token_count:]
            completion_mask = completion_mask[:, -scored_token_count:]

        completions_text = self.prompt_tokenizer.batch_decode(completion_ids_list, skip_special_tokens=True)

        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_completion_tokens = agg_completion_lengths.sum()
        scored_completion_tokens = self.accelerator.gather(completion_mask.sum(dim=1)).sum()
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions_text, completion_ids_list)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for idx, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(torch.nanmean(rewards_per_func[:, idx]).item())
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(nanstd(rewards_per_func[:, idx]).item())

        rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(gather_object(rendered_prompts))
        self._logs["completion"].extend(gather_object(completions_text))
        for idx, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, idx].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())
        self._append_trace_logs(inputs, completions_text, rewards_per_func, advantages)

        output = {
            "prompt_ids": prompt_ids.to(device),
            "prompt_mask": prompt_mask.to(device),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": scored_completion_tokens,
            "pixel_values": generation_inputs["pixel_values"],
            "tgt_sizes": generation_inputs["tgt_sizes"],
            "image_bound": generation_inputs["image_bound"],
            "audio_features": generation_inputs["audio_features"],
            "audio_feature_lens": generation_inputs["audio_feature_lens"],
            "audio_bounds": generation_inputs["audio_bounds"],
            "spk_bounds": generation_inputs["spk_bounds"],
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    @profiling_decorator
    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            tgt_sizes=inputs.get("tgt_sizes"),
            image_bound=inputs.get("image_bound"),
            audio_features=inputs.get("audio_features"),
            audio_feature_lens=inputs.get("audio_feature_lens"),
            audio_bounds=inputs.get("audio_bounds"),
            spk_bounds=inputs.get("spk_bounds"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        per_token_kl = None
        if self.beta != 0.0:
            ref_per_token_logps = inputs.get("ref_per_token_logps")
            if ref_per_token_logps is not None:
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )
                if self.args.use_bias_correction_kl:
                    per_token_kl = per_token_kl * coef_1

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == "luspo":
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * mask).sum() / completion_token_count

        if self.beta != 0.0 and per_token_kl is not None:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss
