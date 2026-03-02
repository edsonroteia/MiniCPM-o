import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoProcessor
from trl import GRPOConfig

from grpo_dataset_av import build_avqa_grpo_dataset
from grpo_rewards import resolve_reward_functions
from grpo_trainer_av import MiniCPMOAVGRPOTrainer

local_rank = 0


@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(default="openbmb/MiniCPM-o-4_5")
    sft_adapter_path: str = field(
        default=(
            "/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/"
            "output_minicpmo45_av_lora_reasoning_full_8gpu_1epoch_video"
        )
    )
    merge_sft_adapter: bool = field(default=True)
    use_lora_for_rl: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default="/weka/kuehne/kqr867/datasets/AVQA/train_qa_cleaned.data")
    eval_data_path: Optional[str] = field(default=None)
    max_samples: Optional[int] = field(default=20)
    max_eval_samples: Optional[int] = field(default=None)
    num_video_frames: int = field(default=8)
    max_slice_nums: int = field(default=1)
    reward_functions: str = field(default="accuracy format reasoning_length")


@dataclass
class TrainingArguments(GRPOConfig):
    output_dir: Optional[str] = field(default="output/output_minicpmo45_av_grpo_smoke")
    learning_rate: float = field(default=1e-6)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.03)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    max_steps: int = field(default=2)
    save_steps: float = field(default=2)
    logging_steps: float = field(default=1)
    save_total_limit: Optional[int] = field(default=2)
    save_only_model: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    remove_unused_columns: bool = field(default=False)
    bf16: Optional[bool] = field(default=False)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    num_generations: int = field(default=4)
    max_completion_length: Optional[int] = field(default=96)
    max_prompt_length: Optional[int] = field(default=2048)
    steps_per_generation: Optional[int] = field(default=4)
    beta: float = field(default=0.04)
    temperature: float = field(default=1.0)
    loss_type: str = field(default="dapo")
    report_to: Optional[str] = field(default="none")
    log_completions: bool = field(default=False)
    wandb_num_logged_completions: int = field(default=8)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir)


def get_parameter_number(model) -> Dict[str, int]:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {"Total": all_param, "Trainable": trainable_params}


def resolve_scale_rewards(loss_type: str, scale_rewards: str) -> str:
    if "--scale_rewards" in sys.argv:
        return scale_rewards
    if loss_type == "dapo":
        return "group"
    if loss_type == "dr_grpo":
        return "none"
    raise ValueError("This minimal GRPO path supports only loss_type=dapo or loss_type=dr_grpo.")


def load_rl_lora_config(sft_adapter_path: str) -> LoraConfig:
    config_path = Path(sft_adapter_path) / "adapter_config.json"
    config = json.loads(config_path.read_text())
    return LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias=config.get("bias", "none"),
        layers_to_transform=config.get("layers_to_transform"),
        modules_to_save=config.get("modules_to_save"),
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.scale_rewards = resolve_scale_rewards(training_args.loss_type, training_args.scale_rewards)
    training_args.reward_weights = None
    training_args.generation_kwargs = dict(training_args.generation_kwargs or {})
    training_args.generation_kwargs.setdefault("min_new_tokens", 1)
    if training_args.logging_dir is None:
        training_args.logging_dir = training_args.output_dir

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    local_rank = training_args.local_rank

    rank0_print(
        f"Loading policy base={model_args.base_model_name_or_path} adapter={model_args.sft_adapter_path} "
        f"use_lora_for_rl={model_args.use_lora_for_rl}"
    )

    if model_args.use_lora_for_rl:
        base_model = AutoModel.from_pretrained(
            model_args.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            init_vision=True,
            init_audio=True,
            init_tts=False,
        )
        processor = AutoProcessor.from_pretrained(
            model_args.base_model_name_or_path,
            trust_remote_code=True,
        )
        if model_args.merge_sft_adapter:
            rank0_print("Merging the SFT LoRA adapter into the base model and attaching a fresh RL LoRA adapter.")
            sft_model = PeftModel.from_pretrained(base_model, model_args.sft_adapter_path, is_trainable=False)
            model = sft_model.merge_and_unload()
            if hasattr(model, "peft_config"):
                delattr(model, "peft_config")
            for param in model.parameters():
                param.requires_grad = False
            model = get_peft_model(model, load_rl_lora_config(model_args.sft_adapter_path))
        else:
            rank0_print("Continuing training directly on the existing SFT LoRA adapter.")
            model = PeftModel.from_pretrained(base_model, model_args.sft_adapter_path, is_trainable=True)
    else:
        rank0_print("Full fine-tuning: loading SFT checkpoint directly (no LoRA).")
        model = AutoModel.from_pretrained(
            model_args.sft_adapter_path,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            init_vision=True,
            init_audio=True,
            init_tts=False,
        )
        processor = AutoProcessor.from_pretrained(
            model_args.sft_adapter_path,
            trust_remote_code=True,
        )
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    train_dataset = build_avqa_grpo_dataset(
        data_path=data_args.data_path,
        num_video_frames=data_args.num_video_frames,
        max_samples=data_args.max_samples,
    )

    if data_args.eval_data_path:
        eval_dataset = build_avqa_grpo_dataset(
            data_path=data_args.eval_data_path,
            num_video_frames=data_args.num_video_frames,
            max_samples=data_args.max_eval_samples,
        )
    else:
        eval_dataset = None
        training_args.do_eval = False
        training_args.eval_strategy = "no"

    reward_funcs, reward_weights, reward_names = resolve_reward_functions(data_args.reward_functions)
    training_args.reward_weights = reward_weights
    rank0_print(
        f"Using rewards={reward_names} weights={reward_weights} loss_type={training_args.loss_type} "
        f"scale_rewards={training_args.scale_rewards}"
    )

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = MiniCPMOAVGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        num_video_frames=data_args.num_video_frames,
        max_slice_nums=data_args.max_slice_nums,
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
