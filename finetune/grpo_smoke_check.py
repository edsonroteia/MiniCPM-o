import argparse

from transformers import AutoProcessor

from grpo_dataset_av import build_avqa_grpo_dataset
from grpo_rewards import resolve_reward_functions
from grpo_trainer_av import prepare_grpo_prompt_example


def main():
    parser = argparse.ArgumentParser(description="Minimal smoke checks for the MiniCPM-o AV GRPO path.")
    parser.add_argument(
        "--data_path",
        default="/weka/kuehne/kqr867/datasets/AVQA/train_qa_cleaned.data",
        help="AVQA JSONL metadata used for RL training.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="openbmb/MiniCPM-o-4_5",
        help="Model or processor source for multimodal tokenization checks.",
    )
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_slice_nums", type=int, default=1)
    parser.add_argument(
        "--skip_processor",
        action="store_true",
        help="Skip loading the MiniCPM processor and only validate dataset + rewards.",
    )
    args = parser.parse_args()

    dataset = build_avqa_grpo_dataset(
        data_path=args.data_path,
        num_video_frames=args.num_video_frames,
        max_samples=args.max_samples,
    )
    print(f"loaded_rows={len(dataset)}")
    row = dataset[0]
    print(f"sample_id={row['sample_id']}")
    print(f"video_path={row['video_path']}")
    print(f"audio_path={row['audio_path']}")
    print(f"solution={row['solution']}")

    reward_funcs, reward_weights, reward_names = resolve_reward_functions("accuracy format reasoning_length")
    completion = f"<think>This clip likely supports option {row['answer_letter']}.</think><answer>{row['answer_letter']}</answer>"
    completions = [completion]
    reward_kwargs = {
        "solution": [row["solution"]],
        "multi_choice": [row["multi_choice"]],
    }
    reward_values = {}
    for name, func in zip(reward_names, reward_funcs, strict=True):
        reward_values[name] = func(
            prompts=[row["prompt"]],
            completions=completions,
            completion_ids=[[]],
            **reward_kwargs,
        )[0]
    print(f"reward_names={reward_names}")
    print(f"reward_weights={reward_weights}")
    print(f"reward_values={reward_values}")

    if args.skip_processor:
        return

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    prepared = prepare_grpo_prompt_example(
        row=row,
        processor=processor,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_slice_nums=args.max_slice_nums,
        num_video_frames=args.num_video_frames,
    )
    print(f"prompt_tokens={prepared['prompt_input_ids'].shape[0]}")
    print(f"image_slots={len(prepared['pixel_values'])}")
    if hasattr(prepared["audio_features"], "shape"):
        print(f"audio_features_shape={tuple(prepared['audio_features'].shape)}")
    else:
        print("audio_features_shape=None")
    if hasattr(prepared["audio_bounds"], "shape"):
        print(f"audio_bounds_shape={tuple(prepared['audio_bounds'].shape)}")
    else:
        print("audio_bounds_shape=None")


if __name__ == "__main__":
    main()
