import argparse
import json

import torch
from transformers import AutoModel, AutoProcessor

from dataset_av import AVSupervisedDataset, av_data_collator


def move_batch_to_cuda(batch):
    batch["input_ids"] = batch["input_ids"].cuda()
    batch["position_ids"] = batch["position_ids"].cuda()
    batch["attention_mask"] = batch["attention_mask"].cuda()

    if hasattr(batch["audio_features"], "cuda"):
        batch["audio_features"] = batch["audio_features"].cuda().to(dtype=torch.bfloat16)

    for key in ("audio_feature_lens", "image_bound", "audio_bounds", "spk_bounds", "tgt_sizes"):
        batch[key] = [value.cuda() if hasattr(value, "cuda") else value for value in batch[key]]

    pixel_values = []
    for sample in batch["pixel_values"]:
        if isinstance(sample, list):
            pixel_values.append([image.cuda().to(dtype=torch.bfloat16) for image in sample])
        else:
            pixel_values.append(sample.cuda().to(dtype=torch.bfloat16))
    batch["pixel_values"] = pixel_values
    return batch


def main():
    parser = argparse.ArgumentParser(description="Run a one-sample forward pass smoke check.")
    parser.add_argument(
        "--model",
        default="openbmb/MiniCPM-o-4_5",
        help="Model name or path.",
    )
    parser.add_argument(
        "--data-path",
        default="sample_data/av_smoke/train.json",
        help="Path to the AV JSON dataset.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Max sequence length for preprocessing.",
    )
    args = parser.parse_args()

    with open(args.data_path, "r") as handle:
        samples = json.load(handle)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    dataset = AVSupervisedDataset(
        samples[:1],
        processor=processor,
        tokenizer=processor.tokenizer,
        max_length=args.max_length,
        max_slice_nums=1,
    )
    batch = av_data_collator([dataset[0]], max_length=args.max_length)

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    ).cuda().eval()

    batch = move_batch_to_cuda(batch)

    with torch.no_grad():
        outputs = model(data=batch, use_cache=False)

    logits = outputs.logits
    print("forward_ok")
    print("logits_shape", tuple(logits.shape))
    print("logits_dtype", logits.dtype)
    print("is_finite", bool(torch.isfinite(logits).all().item()))


if __name__ == "__main__":
    main()
