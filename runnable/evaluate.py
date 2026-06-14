import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dir", type=Path, default=Path("output/results/task_1/multitask"), help="Folder containing best_model.pt."
    )
    p.add_argument("--all-langs", action="store_true", help="Break results down by language (All / EN / ES).")
    p.add_argument(
        "--json-path", type=Path, default=Path("data/exist-memes/training/processed_data.json"),
        help="Training processed_data.json (the val split is carved from this)."
    )
    p.add_argument(
        "--img-dir", type=Path, default=Path("data/exist-memes/training/memes"), help="Training meme images."
    )
    p.add_argument(
        "--tasks", type=str, nargs="+", default=["2.1", "2.2", "2.3"],
        help="Tasks to evaluate (e.g. --tasks 2.1 2.2 2.3)"
    )
    p.add_argument("--seed", type=int, default=42, help="Must match training.")
    p.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train/val split ratio. It must match the original training run (default: 0.8)"
    )
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    p.add_argument("--lora-r", type=int, default=16, help="Must match training.")
    p.add_argument("--lora-alpha", type=int, default=32, help="Must match training.")
    return p.parse_args()


def main() -> None:
    """
    Re-score a trained run on the held-out validation split.

    IMPORTANT: --seed / --lora-r / --lora-alpha / --train_ratio
    must match the training run, or the val split will differ.

    Examples:
      python runnable/evaluate.py --run-dir output/results/task_1/multitask
      python runnable/evaluate.py --run-dir output/results/task_1/multitask --all-langs
    """
    args = parse_args()

    if args.all_langs:
        from exist_2026.evaluate.run_trained_on_val_all_langs import evaluate_run
    else:
        from exist_2026.evaluate.run_trained_on_val_set import evaluate_run

    evaluate_run(
        run_dir=args.run_dir,
        json_path=args.json_path,
        img_dir=args.img_dir,
        tasks=args.tasks,
        seed=args.seed,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
