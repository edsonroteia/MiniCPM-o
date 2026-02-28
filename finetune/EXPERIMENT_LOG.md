# Experiment Log

## 2026-02-28: Full AVQA Reasoning LoRA Run (8 GPU, 1 Epoch)

- Status: running
- Slurm job: `325783`
- Node: `mlcbm004`
- Slurm log: `/weka/kuehne/kqr867/code/MiniCPM-o/slurm-325783.out`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_full_8gpu_1epoch_video`
- W&B project: `https://wandb.ai/edsonroteia/minicpm-av`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/89nbh5xy`

Dataset:
- Train manifest: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/sample_data/av_reasoning_full_video/train.json`
- Eval manifest: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/sample_data/av_reasoning_full_video/eval.json`
- Train samples: `18,279`
- Eval samples: `945`
- Modality format: 8 video frames decoded on the fly from each `.mp4` via `decord`, plus paired audio loaded from the `.wav`
- Target: `solution` field from the HumanOmniV2 reasoning dataset

Training configuration:
- Model: `openbmb/MiniCPM-o-4_5`
- `llm_type`: `qwen`
- LoRA only: `true`
- `tune_vision`: `false`
- `tune_llm`: `false`
- `tune_audio`: `false`
- `init_audio`: `true`
- `max_slice_nums`: `1`
- `model_max_length`: `4096`
- GPUs: `8`
- `per_device_train_batch_size`: `1`
- `per_device_eval_batch_size`: `1`
- `gradient_accumulation_steps`: `4`
- Effective global batch size: `32`
- `max_steps`: `572` (about 1 epoch)
- `eval_steps`: `100`
- `save_steps`: `100`
- `save_total_limit`: `6`
- `dataloader_num_workers`: `4`
- `learning_rate`: `1e-5`
- `weight_decay`: `0.1`
- `adam_beta2`: `0.95`
- `warmup_ratio`: `0.03`
- `lr_scheduler_type`: `cosine`
- `logging_steps`: `10`
- DeepSpeed: disabled

Trainable parameters in this setup:
- LoRA adapters on LLM self-attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `embed_tokens`
- `resampler`
- `audio_projection_layer`

## 2026-02-28: Reasoning Smoke Benchmark

- 1 GPU smoke job: `325769`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_smoke`
- Final metrics: `train_runtime=33.0801s`, `train_steps_per_second=0.605`, `train_samples_per_second=0.605`, `eval_loss=1.8671875`

- 8 GPU smoke job: `325781`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_smoke_8gpu`
- Final metrics: `train_runtime=34.5934s`, `train_steps_per_second=0.578`, `train_samples_per_second=4.625`, `eval_loss=1.875`
- Throughput gain vs 1 GPU: `7.64x` on `train_samples_per_second`

## 2026-02-28: W&B Smoke Run

- Slurm job: `325776`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_smoke_wandb_traces`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/wr3p03ht`
