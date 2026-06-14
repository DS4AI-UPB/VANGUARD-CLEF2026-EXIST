import argparse
from pathlib import Path

from exist_2026.train.train_multitask import train_multitask
from exist_2026.train.train_multitask_ensemble import train_multitask_ensemble
from exist_2026.train.train_single_task import train_single_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--variant", choices=["multitask", "ensemble", "single"], default="multitask", help="Which model to train."
    )
    p.add_argument("--task", choices=["2.1", "2.2", "2.3"], default="2.1", help="Subtask for --variant single.")
    p.add_argument(
        "--json-path", type=Path, default=Path("data/exist-memes/training/processed_data.json"),
        help="Training processed_data.json."
    )
    p.add_argument(
        "--img-dir", type=Path, default=Path("data/exist-memes/training/memes"),
        help="Training meme images."
    )
    p.add_argument(
        "--test-json", type=Path, default=Path("data/exist-memes/test/processed_data.json"),
        help="Test processed_data.json (ignored if it doesn't exist)."
    )
    p.add_argument(
        "--test-img-dir", type=Path, default=Path("data/exist-memes/test/memes"),
        help="Test meme images (ignored if it doesn't exist)."
    )
    p.add_argument(
        "--save-dir", type=Path, default=None,
        help="Where to save checkpoints/logs (default: output/results/task_1/<variant>)."
    )

    p.add_argument(
        "--tasks", nargs="+", default=["2.1", "2.2", "2.3"], choices=["2.1", "2.2", "2.3"],
        help="Subtasks for multitask/ensemble."
    )
    p.add_argument(
        "--train-ratio", type=float, default=0.8, help="Fraction of data used for training (rest goes to validation)."
    )
    p.add_argument(
        "--text-model", type=str, default="FacebookAI/xlm-roberta-base",
        help="HuggingFace model ID for the text encoder.",
    )
    p.add_argument(
        "--image-model", type=str, default="openai/clip-vit-base-patch32",
        help="HuggingFace model ID for the image encoder.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main() -> None:
    """
    Train an VANGUARD EXIST 2026 model.

    Defaults to the multitask model (all three subtasks, no ensemble) - the paper's
    best All-split configuration. Run after preprocess.py.

    Examples:
      python runnable/train.py
      python runnable/train.py --variant ensemble
      python runnable/train.py --variant single --task 2.1
      python runnable/train.py --num-epochs 30 --lora-r 8
    """
    args = parse_args()

    # Use the test set only if it's actually present.
    test_json = args.test_json if args.test_json.is_file() else None
    test_img_dir = args.test_img_dir if args.test_img_dir.is_dir() else None
    if test_json is None or test_img_dir is None:
        print("[note] No test set found — training/validating only.")

    save_dir = args.save_dir or Path("output/results/task_1") / args.variant

    if args.variant == "single":
        train_single_task(
            task=args.task,
            json_path=args.json_path,
            img_dir=args.img_dir,
            test_json=test_json,
            test_img_dir=test_img_dir,
            save_dir=save_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            num_epochs=args.num_epochs,
            text_model=args.text_model,
            image_model=args.image_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
    else:
        train_fn = train_multitask if args.variant == "multitask" else train_multitask_ensemble
        train_fn(
            json_path=args.json_path,
            img_dir=args.img_dir,
            test_json=test_json,
            test_img_dir=test_img_dir,
            save_dir=save_dir,
            tasks=set(args.tasks),
            seed=args.seed,
            train_ratio=args.train_ratio,
            num_epochs=args.num_epochs,
            text_model=args.text_model,
            image_model=args.image_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )


if __name__ == "__main__":
    main()
