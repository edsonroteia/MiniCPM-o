import json
import logging
import os
from importlib.util import find_spec
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from types import MethodType

import torch
import transformers
from accelerate.utils import DistributedType
from transformers import AutoModel, AutoProcessor, TrainerCallback
from transformers.integrations import deepspeed

from dataset_av import AVSupervisedDataset, av_data_collator
from trainer import CPMTrainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-o-4_5")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length."},
    )
    tune_vision: Optional[bool] = field(default=False)
    tune_llm: Optional[bool] = field(default=False)
    tune_audio: Optional[bool] = field(default=False)
    init_audio: Optional[bool] = field(default=True)
    llm_type: str = field(default="qwen")
    use_lora: Optional[bool] = field(default=True)
    max_slice_nums: Optional[int] = field(default=1)
    wandb_num_eval_examples: int = field(
        default=2,
        metadata={"help": "Number of evaluation examples to log to W&B as prediction traces."},
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


local_rank = 0
logger = logging.getLogger(__name__)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir)


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {"Total": all_param, "Trainable": trainable_params}


def make_supervised_data_module(processor, tokenizer, data_args, max_length=4096, max_slice_nums=1) -> Dict:
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = AVSupervisedDataset(
        train_json,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        max_slice_nums=max_slice_nums,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = AVSupervisedDataset(
            eval_json,
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            max_slice_nums=max_slice_nums,
        )
    else:
        eval_dataset = None

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": partial(av_data_collator, max_length=max_length),
    }


def report_to_includes(report_to: Any, target: str) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        values = [value.strip() for value in report_to.split(",") if value.strip()]
        return target in values
    if isinstance(report_to, (list, tuple, set)):
        return target in report_to
    return False


class WandbPredictionLoggerCallback(TrainerCallback):
    """Logs teacher-forced prediction traces to W&B after evaluation."""

    def __init__(self, dataset, data_collator, tokenizer, use_lora: bool, num_examples: int = 2):
        self.dataset = dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.use_lora = use_lora
        self.num_examples = max(0, num_examples)
        self._wandb = None

    def _get_wandb(self):
        if self._wandb is not None:
            return self._wandb

        if find_spec("wandb") is None:
            logger.warning("W&B example logging requested, but wandb is not installed.")
            self._wandb = False
            return None

        import wandb

        self._wandb = wandb
        return wandb

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

    def _unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _run_forward(self, model, batch):
        model_inputs = dict(batch)
        model_inputs.pop("labels", None)

        if self.use_lora:
            with model._enable_peft_forward_hooks(**model_inputs):
                return model.base_model(data=model_inputs, use_cache=False)
        return model(data=model_inputs, use_cache=False)

    def _decode_prediction(self, logits: torch.Tensor, labels: torch.Tensor) -> str:
        mask = labels != -100
        if not torch.any(mask):
            return ""

        pred_ids = logits[mask].argmax(dim=-1)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            eos_positions = (pred_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                pred_ids = pred_ids[: int(eos_positions[0].item())]

        decoded = self.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=False)
        return decoded.strip()

    def _build_prompt_text(self, sample: Dict[str, Any]) -> str:
        parts = []
        for message in sample.get("conversations", []):
            if message.get("role") == "user":
                parts.append(message.get("content", ""))
        return "\n\n".join(part for part in parts if part)

    def _log_prediction_examples(self, args, state, model):
        if self.num_examples <= 0:
            return
        if not state.is_world_process_zero:
            return
        if self.dataset is None or len(self.dataset) == 0:
            return

        wandb = self._get_wandb()
        if not wandb or getattr(wandb, "run", None) is None:
            return

        unwrapped_model = self._unwrap_model(model)
        try:
            device = next(unwrapped_model.parameters()).device
        except StopIteration:
            return

        table = wandb.Table(
            columns=[
                "global_step",
                "sample_id",
                "num_frames",
                "prompt",
                "target",
                "prediction",
                "audio_path",
            ]
        )

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for idx in range(min(self.num_examples, len(self.dataset))):
                    raw_sample = self.dataset.raw_data[idx]
                    example = self.dataset[idx]
                    batch = self.data_collator([example])
                    labels = batch["labels"][0].detach().cpu()
                    batch = self._move_to_device(batch, device)
                    outputs = self._run_forward(unwrapped_model, batch)
                    prediction = self._decode_prediction(outputs.logits[0].detach().cpu(), labels)

                    image_spec = raw_sample.get("image")
                    if isinstance(image_spec, dict):
                        num_frames = len(image_spec)
                    elif image_spec:
                        num_frames = 1
                    else:
                        num_frames = 0

                    target = ""
                    for message in raw_sample.get("conversations", []):
                        if message.get("role") == "assistant":
                            target = message.get("content", "")
                            break

                    table.add_data(
                        state.global_step,
                        raw_sample.get("id", idx),
                        num_frames,
                        self._build_prompt_text(raw_sample),
                        target,
                        prediction,
                        raw_sample.get("audio", ""),
                    )
        except Exception:
            logger.exception("Failed to log W&B prediction traces.")
            return
        finally:
            if was_training:
                model.train()

        wandb.log({"eval/prediction_examples": table})

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._log_prediction_examples(args, state, model)
        return control


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
        init_vision=True,
        init_audio=training_args.init_audio,
        init_tts=False,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    if not training_args.tune_vision and hasattr(model, "vpm"):
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm and hasattr(model, "llm"):
        model.llm.requires_grad_(False)
    if not training_args.tune_audio and hasattr(model, "apm"):
        model.apm.requires_grad_(False)

    if training_args.use_lora:
        if training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")

        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-o model.")
        for _, param in model.llm.named_parameters():
            param.requires_grad = False

        modules_to_save = ["embed_tokens", "resampler", "audio_projection_layer"]
        if training_args.tune_vision and hasattr(model, "vpm"):
            modules_to_save.append("vpm")
        if training_args.tune_audio and hasattr(model, "apm"):
            modules_to_save.append("apm")

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, "get_input_embeddings"):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))
    rank0_print(f"llm_type={training_args.llm_type}")

    data_module = make_supervised_data_module(
        processor=processor,
        tokenizer=tokenizer,
        data_args=data_args,
        max_length=training_args.model_max_length,
        max_slice_nums=training_args.max_slice_nums,
    )

    callbacks = []
    if report_to_includes(training_args.report_to, "wandb"):
        callbacks.append(
            WandbPredictionLoggerCallback(
                dataset=data_module["eval_dataset"] or data_module["train_dataset"],
                data_collator=data_module["data_collator"],
                tokenizer=tokenizer,
                use_lora=bool(training_args.use_lora),
                num_examples=training_args.wandb_num_eval_examples,
            )
        )

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = CPMTrainer(
        model=model,
        tokenizer=processor,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
