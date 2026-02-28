import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "sample_data" / "av_smoke"
DEFAULT_AVQA_ROOT = Path("/weka/kuehne/kqr867/datasets/AVQA")
DEFAULT_AVQA_METADATA = DEFAULT_AVQA_ROOT / "train_qa_cleaned.data"
DEFAULT_NUM_FRAMES = 8
DEMO_VIDEO = REPO_ROOT / "assets" / "demo_video.mp4"


def run_command(args):
    subprocess.run(args, check=True)


def normalize_existing_path(path_str: str) -> Path:
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


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return float(result.stdout.strip())


def extract_frame(video_path: Path, image_path: Path, timestamp: float) -> None:
    timestamp = max(timestamp, 0.0)
    run_command(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-q:v",
            "2",
            str(image_path),
        ]
    )


def extract_evenly_spaced_frames(video_path: Path, output_dir: Path, stem: str, num_frames: int):
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    duration = probe_duration(video_path)
    if duration <= 0:
        timestamps = [0.0] * num_frames
    else:
        timestamps = [duration * (idx + 1) / (num_frames + 1) for idx in range(num_frames)]

    frame_paths = []
    for idx, timestamp in enumerate(timestamps):
        image_path = output_dir / f"{stem}_{idx:02d}.jpg"
        extract_frame(video_path, image_path, timestamp)
        frame_paths.append(image_path)

    return frame_paths


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
    )


def build_image_placeholders(num_frames: int):
    return [f"<image_{idx:02d}>" for idx in range(num_frames)]


def build_mcq_prompt(record: dict, num_frames: int) -> str:
    letters = ["A", "B", "C", "D"]
    lines = build_image_placeholders(num_frames) + ["<audio>", record["question_text"]]
    for idx, option in enumerate(record["multi_choice"]):
        lines.append(f"{letters[idx]}. {option}")
    return "\n".join(lines)


def build_mcq_answer(record: dict) -> str:
    letter = ["A", "B", "C", "D"][int(record["answer"])]
    return f"The answer is {letter}."


def load_avqa_records(metadata_path: Path, limit: int) -> list:
    if metadata_path.suffix == ".data":
        records = []
        with open(metadata_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
        return records

    with open(metadata_path, "r") as handle:
        records = json.load(handle)
    return records[:limit]


def resolve_avqa_video_path(avqa_root: Path, record: dict) -> Path:
    audio_path = normalize_existing_path(record["audio_path"])
    split = "eval" if "/eval/" in audio_path.as_posix() else "train"
    video_path = avqa_root / split / f"{record['video_name']}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing AVQA video: {video_path}")
    return video_path


def prepare_from_avqa(avqa_root: Path, metadata_path: Path, output_dir: Path, count: int, num_frames: int) -> tuple:
    records = load_avqa_records(metadata_path, count)
    samples = []
    for idx, record in enumerate(records):
        audio_path = normalize_existing_path(record["audio_path"])
        video_path = resolve_avqa_video_path(avqa_root, record)
        frame_paths = extract_evenly_spaced_frames(video_path, output_dir, f"avqa_{idx:03d}", num_frames)
        placeholders = build_image_placeholders(num_frames)
        samples.append(
            {
                "id": str(record["id"]),
                "image": {
                    placeholder: str(frame_path.resolve())
                    for placeholder, frame_path in zip(placeholders, frame_paths)
                },
                "audio": str(audio_path.resolve()),
                "conversations": [
                    {
                        "role": "user",
                        "content": build_mcq_prompt(record, num_frames),
                    },
                    {
                        "role": "assistant",
                        "content": build_mcq_answer(record),
                    },
                ],
            }
        )

    if not samples:
        raise ValueError("No AVQA samples were prepared.")

    return samples, samples[: min(4, len(samples))]


def prepare_from_demo(output_dir: Path, num_frames: int) -> tuple:
    if not DEMO_VIDEO.exists():
        raise FileNotFoundError(f"Missing demo video: {DEMO_VIDEO}")

    frame_paths = extract_evenly_spaced_frames(DEMO_VIDEO, output_dir, "demo_video", num_frames)
    audio_path = output_dir / "demo_video.wav"
    extract_audio(DEMO_VIDEO, audio_path)
    placeholders = build_image_placeholders(num_frames)

    sample = {
        "id": "demo_video",
        "image": {
            placeholder: str(frame_path.resolve())
            for placeholder, frame_path in zip(placeholders, frame_paths)
        },
        "audio": str(audio_path.resolve()),
        "conversations": [
            {
                "role": "user",
                "content": "\n".join(placeholders + ["<audio>", "Summarize this clip in one sentence."]),
            },
            {
                "role": "assistant",
                "content": "This clip shows a short video segment with synchronized environmental audio.",
            },
        ],
    }
    return [sample], [sample]


def write_json(path: Path, payload: list) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare a small audio-visual fine-tuning dataset.")
    parser.add_argument(
        "--source",
        choices=["auto", "avqa", "demo"],
        default="auto",
        help="Source dataset to use.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of AVQA samples to prepare.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames to extract per sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extracted frames and dataset JSON files.",
    )
    parser.add_argument(
        "--avqa-root",
        type=Path,
        default=DEFAULT_AVQA_ROOT,
        help="Root directory of the AVQA dataset.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_AVQA_METADATA,
        help="AVQA metadata file (.json or .data JSONL).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "demo":
        train_samples, eval_samples = prepare_from_demo(args.output_dir, args.num_frames)
    else:
        try:
            if args.source == "avqa" or args.avqa_root.exists():
                train_samples, eval_samples = prepare_from_avqa(
                    args.avqa_root,
                    args.metadata_path,
                    args.output_dir,
                    args.count,
                    args.num_frames,
                )
            else:
                raise FileNotFoundError("AVQA root is not available.")
        except Exception:
            if args.source == "avqa":
                raise
            train_samples, eval_samples = prepare_from_demo(args.output_dir, args.num_frames)

    write_json(args.output_dir / "train.json", train_samples)
    write_json(args.output_dir / "eval.json", eval_samples)
    print(f"Wrote {len(train_samples)} train samples to {(args.output_dir / 'train.json').resolve()}")
    print(f"Wrote {len(eval_samples)} eval samples to {(args.output_dir / 'eval.json').resolve()}")


if __name__ == "__main__":
    main()
