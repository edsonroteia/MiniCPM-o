# Experiment Log

## 2026-03-02: Full-FT Pipeline — SFT Ablations + GRPO Full Fine-Tuning

Sequential pipeline on mlcbm004 (8x H100), chained via `sbatch --dependency`.

### SFT-A: Higher Learning Rate (lr=1e-5)

- Status: ⏳ pending (chained after baseline job 326475)
- Slurm job: `326479`
- Slurm script: `submit_finetune_av_fullft_sftA.slurm`
- Output directory: `output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_1e5`
- W&B run name: `fullft-8f-1e5-8gpu`
- Rationale: 1e-6 may be too conservative for full FT; LoRA run (325783) used 1e-5 successfully
- Config: same as baseline except `LEARNING_RATE=1e-5`

### SFT-B: Tune LLM + Audio Encoder (lr=1e-6)

- Status: ⏳ pending (chained after SFT-A)
- Slurm job: `326480`
- Slurm script: `submit_finetune_av_fullft_sftB.slurm`
- Output directory: `output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_llm_audio`
- W&B run name: `fullft-8f-1e6-llm-audio-8gpu`
- Rationale: MMAU/MMAR performance flat; audio encoder may be a bottleneck
- Config: same as baseline except `TUNE_AUDIO=true`

### SFT-C: Tune LLM + Vision + Audio (lr=1e-6)

- Status: ⏳ pending (chained after SFT-B)
- Slurm job: `326481`
- Slurm script: `submit_finetune_av_fullft_sftC.slurm`
- Output directory: `output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_all`
- W&B run name: `fullft-8f-1e6-llm-vision-audio-8gpu`
- Rationale: unfreeze all encoders for co-adaptation; tests whether vision tuning adds value over SFT-B
- Config: same as baseline except `TUNE_VISION=true`, `TUNE_AUDIO=true`

### SFT-D: Text Answers — 50% choice text instead of letter (lr=1e-6)

- Status: ⏳ pending (chained after SFT-C)
- Slurm job: `326483`
- Slurm script: `submit_finetune_av_fullft_sftD.slurm`
- Output directory: `output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_textanswer`
- W&B run name: `fullft-8f-1e6-textanswer-8gpu`
- Rationale: rigid `<answer>LETTER</answer>` training hurts on content-match benchmarks; mixed letter/text targets teach the model both answer formats
- Config: same as baseline except uses preprocessed dataset where ~50% of `<answer>C</answer>` are replaced with `<answer>sound of wind</answer>` (the actual choice text)
- Data: `sample_data/av_reasoning_full_video/train_textanswer.json` (9,085/18,279 converted)
- Preprocessing: `preprocess_text_answers.py --fraction 0.5 --seed 42`

### GRPO Full Fine-Tuning (on SFT baseline lr=1e-6 checkpoint)

- Status: ⏳ pending (chained after SFT-D)
- Slurm job: `326482`
- Slurm script: `submit_finetune_grpo_av_fullft.slurm`
- Output directory: `output/output_minicpmo45_av_grpo_fullft`
- W&B run name: `grpo-fullft-8gpu`
- Rationale: full-FT GRPO avoids LoRA capacity bottleneck; uses best SFT baseline checkpoint
- Key config:
  - `USE_LORA_FOR_RL=false` (new code path)
  - `LEARNING_RATE=5e-7` (conservative for full-FT RL)
  - `BETA=0.0` (no KL penalty — no ref model for full-FT)
  - `REWARD_FUNCTIONS="accuracy format"` (drop reasoning_length — poorly calibrated)
  - `MAX_COMPLETION_LENGTH=384` (96 clips 100%; 256-tok run showed mean=142)
  - `NUM_GENERATIONS=4`, `GRADIENT_ACCUMULATION_STEPS=4` (effective batch=32)
  - `LOSS_TYPE=dapo` (removes lower clip for better exploration)
  - `MAX_STEPS=200` (~5 hours estimated)

### GRPO Full Fine-Tuning on SFT-D (text answers checkpoint)

- Status: ⏳ pending (chained after GRPO baseline)
- Slurm job: `326487`
- Slurm script: `submit_finetune_grpo_av_fullft_sftD.slurm`
- Output directory: `output/output_minicpmo45_av_grpo_fullft_sftD`
- W&B run name: `grpo-fullft-sftD-8gpu`
- SFT source: highest checkpoint from `output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_textanswer`
- Config: same as baseline GRPO (lr=5e-7, beta=0, accuracy=0.8/format=0.2, 200 steps)

### GRPO Full Fine-Tuning on SFT-C (LLM+vision+audio checkpoint)

- Status: ⏳ pending (chained after GRPO-D)
- Slurm job: `326488`
- Slurm script: `submit_finetune_grpo_av_fullft_sftC.slurm`
- Output directory: `output/output_minicpmo45_av_grpo_fullft_sftC`
- W&B run name: `grpo-fullft-sftC-8gpu`
- SFT source: highest checkpoint from `output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_all`
- Config: same as baseline GRPO (lr=5e-7, beta=0, accuracy=0.8/format=0.2, 200 steps)

Code changes for GRPO full-FT:
- `finetune_grpo_av.py`: added `use_lora_for_rl` flag; full-FT loads SFT checkpoint directly
- `finetune_grpo_av.sh`: passes `USE_LORA_FOR_RL` and `MERGE_SFT_ADAPTER` env vars
- `grpo_trainer_av.py`: guards KL reference path — skips KL when no peft_config (full-FT)
- `grpo_rewards.py`: format_reward now accepts any non-empty answer (not just [A-D]); accuracy weight 0.8, format weight 0.2

Pipeline script: `submit_fullft_pipeline.sh [baseline_job_id]`

---

## 2026-03-02: Full Fine-Tuning SFT (8 GPU, 1 Epoch, 8 Frames, lr=1e-6)

- Status: 🔄 running
- Slurm job: `326460` (originally `326458`, resubmitted with 1-day time limit)
- Node: `mlcbm004`
- Slurm log: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/slurm-326460.out`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_1e6`
- W&B project: `https://wandb.ai/edsonroteia/minicpm-av`
- W&B run name: `fullft-8f-1e6-8gpu`

Dataset:
- Train manifest: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/sample_data/av_reasoning_full_video/train.json`
- Eval manifest: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/sample_data/av_reasoning_full_video/eval.json`
- Train samples: `18,279`
- Eval samples: `945`
- Modality format: 8 video frames decoded on the fly from each `.mp4` via `decord`, plus paired audio loaded from the `.wav`

Training configuration:
- Model: `openbmb/MiniCPM-o-4_5`
- `llm_type`: `qwen`
- **Full fine-tuning** (no LoRA): `use_lora=false`, `tune_llm=true`
- `tune_vision`: `false`
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
- `learning_rate`: `1e-6`
- `weight_decay`: `0.1`
- `adam_beta2`: `0.95`
- `warmup_ratio`: `0.03`
- `lr_scheduler_type`: `cosine`
- `logging_steps`: `10`
- DeepSpeed: ZeRO-2 (`ds_config_zero2.json`)

Compared to LoRA run (job 325783):
- LoRA disabled, all LLM parameters trainable
- Learning rate reduced from `1e-5` to `1e-6`
- DeepSpeed ZeRO-2 enabled for memory sharding

## 2026-03-02: Full Fine-Tuning SFT (8 GPU, 1 Epoch, 8 Frames, lr=5e-6)

- Status: ⏳ pending
- Slurm job: `326465` (resubmitted: 4 GPUs on mlcbm005 via reservation, pending node availability)
- Node: `mlcbm005`
- Slurm log: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/slurm-326465.out`
- GPUs: `4` (with `gradient_accumulation_steps=8` to keep effective batch size 32)
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_fullft_reasoning_full_8gpu_1epoch_video_5e6`
- W&B project: `https://wandb.ai/edsonroteia/minicpm-av`
- W&B run name: `fullft-8f-5e6-8gpu`

Dataset:
- Same as the `lr=1e-6` run above

Training configuration:
- Same as the `lr=1e-6` run above, except:
- `learning_rate`: `5e-6`

## 2026-02-28: Full AVQA Reasoning LoRA Run (8 GPU, 1 Epoch, 8 Frames)

- Status: ✅ completed
- Slurm job: `325783`
- Node: `mlcbm004`
- Slurm log: `/weka/kuehne/kqr867/code/MiniCPM-o/slurm-325783.out`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_full_8gpu_1epoch_video`
- W&B project: `https://wandb.ai/edsonroteia/minicpm-av`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/89nbh5xy`
- Final metrics: `train_runtime=1639.9767s`, `train_steps_per_second=0.349`, `train_samples_per_second=11.161`, `train_loss=1.2134307647918487`, `final_eval_loss=1.1241484880447388`

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

## 2026-02-28: Full AVQA Reasoning LoRA Run (8 GPU, 1 Epoch, 16 Frames)

- Status: ✅ completed
- Slurm job: `325791`
- Node: `mlcbm004`
- Slurm log: `/weka/kuehne/kqr867/code/MiniCPM-o/slurm-325791.out`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_full_16f_8gpu_1epoch_video_mlcbm004`
- W&B project: `https://wandb.ai/edsonroteia/minicpm-av`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/wzch5h4u`
- Final metrics: `train_runtime=1943.7282s`, `train_steps_per_second=0.294`, `train_samples_per_second=9.417`, `train_loss=1.2113909954791302`

Dataset:
- Train manifest: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/sample_data/av_reasoning_full_video_16f/train.json`
- Eval manifest: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/sample_data/av_reasoning_full_video_16f/eval.json`
- Train samples: `18,279`
- Eval samples: `945`
- Modality format: 16 video frames decoded on the fly from each `.mp4` via `decord`, plus paired audio loaded from the `.wav`
- Target: `solution` field from the HumanOmniV2 reasoning dataset

Training configuration:
- Same as the completed 8-frame full run above, except the prompts request `16` frames instead of `8`
- GPUs: `8`
- Effective global batch size: `32`
- `max_steps`: `572` (about 1 epoch)
- `dataloader_num_workers`: `4`

Scheduler note:
- Initial submission `325786` was canceled after it remained pending with reason `BeginTime`
- Resubmission `325789` briefly entered startup, then was requeued by Slurm and sent back to `BeginTime`
- Final retry `325790` was submitted with `--no-requeue` and was then cancelled by Slurm in `Prolog` after 6 seconds
- Those failures were specific to `mlcbm005`; the successful retry ran on `mlcbm004` as job `325791`

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

## 2026-02-28: GRPO Smoke Validation (TRL, MiniCPM-o-4.5 from SFT LoRA)

Policy initialization:
- Base model: `openbmb/MiniCPM-o-4_5`
- SFT adapter source: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_lora_reasoning_full_8gpu_1epoch_video`
- Adapter type: LoRA (`r=64`, `alpha=64`, `dropout=0.05`) plus saved modules `embed_tokens`, `resampler`, `audio_projection_layer`

Initial 8-GPU TRL smoke runs:
- DAPO smoke: job `325846`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_grpo_smoke_8gpu_dapo_sftprompt_wandb`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/xcd64ryt`
- Final metrics: `train_runtime=13.0651s`, `train_steps_per_second=0.153`, `train_samples_per_second=1.225`, `train_loss=0.24750074744224548`

- Dr. GRPO smoke: job `325847`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_grpo_smoke_8gpu_drgrpo_sftprompt_wandb`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/ukpxnm0h`
- Final metrics: `train_runtime=13.2542s`, `train_steps_per_second=0.151`, `train_samples_per_second=1.207`, `train_loss=0.10305114835500717`

Prompt-format fix:
- Root cause: the GRPO chat-template path was calling `apply_chat_template(..., enable_thinking=False)`, which forced `<|im_start|>assistant\n<think>\n\n</think>\n\n` before generation
- Fix: remove `enable_thinking=False` so the assistant prompt matches the SFT behavior and the model generates the reasoning block itself
- Custom `grpo_traces` W&B table now logs the raw SFT-style prompt (`<image_00> ... <image_07>`, `<audio>`, question, choices) instead of only the rendered MiniCPM markup

Prompt-fix smoke checks:
- 96-token smoke: job `325871`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_grpo_smoke_promptfix_wandb_96tok`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/h37drd69`
- Final metrics: `train_runtime=18.1826s`, `train_steps_per_second=0.055`, `train_samples_per_second=0.055`, `train_loss=0.0009528001537546515`
- Result: `completions/clipped_ratio=1.0`; every completion hit the `96`-token cap and rewards stayed at `0.0`

- 256-token smoke: job `325872`
- Output directory: `/weka/kuehne/kqr867/code/MiniCPM-o/finetune/output/output_minicpmo45_av_grpo_smoke_promptfix_wandb_256tok`
- W&B run: `https://wandb.ai/edsonroteia/minicpm-av/runs/hs7xeluy`
- Final metrics: `train_runtime=21.2681s`, `train_steps_per_second=0.047`, `train_samples_per_second=0.047`, `train_loss=0.003031316213309765`
- Key GRPO metrics: `completions/mean_length=142.5`, `completions/min_length=135.0`, `completions/max_length=150.0`, `completions/clipped_ratio=0.0`
- Reward summary: `accuracy_reward=1.0`, `format_reward=0.25`, `reasoning_length_reward=0.0905534029006958`, aggregate `reward=1.3405535221099854`
- Result: completions now reach `</think>` cleanly, but they still end with plain `A` instead of the fully formatted `<answer>A</answer>` suffix

---

## Changelog

- **2026-03-02**: Submitted full-FT pipeline on mlcbm004: 4 SFT ablations (A=lr1e-5, B=LLM+audio, C=all encoders, D=text answers) + 3 GRPO full-FT runs (baseline/SFT-D/SFT-C). Jobs 326479-326488. Code changes: full-FT GRPO path, lenient format reward, adjusted reward weights (accuracy=0.8/format=0.2).
