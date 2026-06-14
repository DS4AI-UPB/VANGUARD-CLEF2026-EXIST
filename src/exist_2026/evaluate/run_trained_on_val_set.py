import argparse
import json
from pathlib import Path

import joblib
import torch

from exist_2026.evaluate.eval_multitask import evaluate_epoch_all
from exist_2026.path_manager import PathManager
from exist_2026.train.helpers import (
    build_multitask_dataloaders,
    build_multitask_model,
    get_raw_data,
    compute_true_2_1_ratios,
    find_optimal_threshold,
)
from exist_2026.train.nn.determinism import seed_everything
from exist_2026.train.nn.losses import MultitaskLoss
from exist_2026.train.train_steps import val_step


def evaluate_run(
        run_dir: str | Path,
        json_path: str | Path,
        img_dir: str | Path,
        tasks: set[str] | None = None,
        seed: int = 42,
        train_ratio: float = 0.8,
        batch_size: int = 16,
        text_model: str = "FacebookAI/xlm-roberta-base",
        image_model: str = "openai/clip-vit-base-patch32",
        lora_r: int = 16,
        lora_alpha: int = 32,
) -> dict:
    """
    Load best_model.pt from `run_dir` and recompute all PyEvALL metrics on val.

    The seed, train_ratio, and batch_size MUST match those used during the original training run,
    otherwise the val split will differ.
    """
    tasks = tasks or {"2.1", "2.2", "2.3"}
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    seed_everything(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Tasks: {sorted(tasks)} | Run: {run_dir}")

    train_loader, val_loader, _, scaler, _, _ = build_multitask_dataloaders(
        text_model=text_model, image_model=image_model,
        json_path=Path(json_path), img_dir=Path(img_dir),
        test_json=None, test_img_dir=None,
        tasks=tasks, train_ratio=train_ratio, batch_size=batch_size, seed=seed,
    )

    saved_scaler_path = run_dir / "sensor_scaler.joblib"
    if saved_scaler_path.exists():
        saved_scaler = joblib.load(saved_scaler_path)
        if not (
                (saved_scaler.mean_ == scaler.mean_).all()
                and (saved_scaler.scale_ == scaler.scale_).all()
        ):
            print("WARNING: rebuilt scaler != saved scaler. Using saved one.")
            val_loader.dataset.dataset.scaler = saved_scaler

    model = build_multitask_model(text_model, image_model, device, tasks, lora_r, lora_alpha)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {ckpt_path}")

    criterion = MultitaskLoss(tasks=tasks)

    _, val_preds = val_step(model, val_loader, criterion, device, tasks)
    raw_data = get_raw_data(val_loader)

    threshold_2_1 = 0.5
    if val_preds["probs_2_1"] is not None:
        true_21 = compute_true_2_1_ratios(val_preds["ids"], raw_data)
        threshold_2_1, best_f1 = find_optimal_threshold(true_21, val_preds["probs_2_1"])
        print(f"Recovered 2.1 threshold: {threshold_2_1:.2f} (val F1={best_f1:.4f})")

    thresh_file = run_dir / "best_thresholds.json"
    threshold_2_3 = 0.5
    if thresh_file.exists():
        with open(thresh_file) as f:
            saved = json.load(f)
        threshold_2_1 = saved.get("threshold_2_1", threshold_2_1)
        threshold_2_3 = saved.get("threshold_2_3", threshold_2_3)
        print(f"Using saved thresholds: 2.1={threshold_2_1:.2f}, 2.3={threshold_2_3:.2f}")

    official = evaluate_epoch_all(
        ids=val_preds["ids"],
        probs_2_1=val_preds["probs_2_1"],
        probs_2_2=val_preds["probs_2_2"],
        probs_2_3=val_preds["probs_2_3"],
        dataset_data=raw_data,
        threshold_2_1=threshold_2_1,
        threshold_2_3=threshold_2_3,
    )

    print("\n" + "=" * 60)
    print("FULL VALIDATION METRICS")
    print("=" * 60)
    for task in sorted(tasks):
        print(f"\nTask {task}")
        print("-" * 40)
        for k in sorted(official.keys()):
            if k.startswith(f"{task}/"):
                print(f"\t{k:50s} {official[k]:.4f}")

    out_path = run_dir / "val_metrics_full.json"
    with open(out_path, "w") as f:
        json.dump(
            {k: round(v, 6) for k, v in official.items()},
            f, indent=2,
        )
    print(f"\nSaved -> {out_path}")

    return official


def main():
    DATA = PathManager.DATA_EXIST_DIR

    parser = argparse.ArgumentParser(
        description="Reload a saved checkpoint and recompute full PyEvALL validation metrics."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to the run directory containing best_model.pt"
    )
    parser.add_argument(
        "--json-path", type=str, default=str(DATA / "training" / "processed_data.json"),
        help="Path to the processed data JSON file"
    )
    parser.add_argument(
        "--img-dir", type=str, default=str(DATA / "training" / "memes"), help="Path to the memes image directory"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["2.1", "2.2", "2.3"],
        help="Tasks to evaluate (e.g. --tasks 2.1 2.2 2.3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. It must match the original training run (default: 42)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train/val split ratio. It must match the original training run (default: 0.8)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--text-model", type=str, default="FacebookAI/xlm-roberta-base", help="HuggingFace text encoder model name"
    )
    parser.add_argument(
        "--image-model", type=str, default="openai/clip-vit-base-patch32",
        help="HuggingFace image encoder model name"
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    args = parser.parse_args()

    evaluate_run(
        run_dir=args.run_dir,
        json_path=args.json_path,
        img_dir=args.img_dir,
        tasks=set(args.tasks),
        seed=args.seed,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        text_model=args.text_model,
        image_model=args.image_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
