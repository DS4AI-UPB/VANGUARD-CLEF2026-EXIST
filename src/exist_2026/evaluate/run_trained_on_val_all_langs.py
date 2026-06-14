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


def filter_preds_by_lang(
        val_preds: dict, raw_data: list[dict], lang: str | None
) -> dict:
    """
    Return a new predictions dict containing only items whose `lang` field
    matches `lang`. If lang is None, returns val_preds unchanged.

    The probability lists are kept aligned with the filtered ids.
    """
    if lang is None:
        return val_preds

    id_to_lang = {str(item.get("id_EXIST", "")): item.get("lang") for item in raw_data}
    keep_mask = [id_to_lang.get(str(mid)) == lang for mid in val_preds["ids"]]

    def _filter_list(lst):
        if lst is None:
            return None
        return [v for v, k in zip(lst, keep_mask) if k]

    return {
        "ids": [v for v, k in zip(val_preds["ids"], keep_mask) if k],
        "probs_2_1": _filter_list(val_preds["probs_2_1"]),
        "probs_2_2": _filter_list(val_preds["probs_2_2"]),
        "probs_2_3": _filter_list(val_preds["probs_2_3"]),
    }


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
    Load best_model.pt from `run_dir` and recompute all PyEvALL metrics on val,
    broken down by language (All / EN / ES).

    The seed, train_ratio, and batch_size MUST match those used during the
    original training run, otherwise the val split will differ.
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

    results_by_lang = {}
    for label, lang_filter in [("All", None), ("EN", "en"), ("ES", "es")]:
        subset = filter_preds_by_lang(val_preds, raw_data, lang_filter)
        n = len(subset["ids"])
        print(f"\nEvaluating split '{label}' ({n} items)...")

        if n == 0:
            print(f"  No items match lang={lang_filter}, skipping.")
            continue

        official = evaluate_epoch_all(
            ids=subset["ids"],
            probs_2_1=subset["probs_2_1"],
            probs_2_2=subset["probs_2_2"],
            probs_2_3=subset["probs_2_3"],
            dataset_data=raw_data,
            threshold_2_1=threshold_2_1,
            threshold_2_3=threshold_2_3,
        )
        results_by_lang[label] = official

    print("\n" + "=" * 70)
    print("FULL VALIDATION METRICS (paste these into Tables 4 and 5)")
    print("=" * 70)
    for task in sorted(tasks):
        print(f"\n----- Task {task} -----")
        print(f"  {'Split':<5} | {'ICMSoft':>10} | {'ICMSoftNorm':>12} | {'CrossEntropy':>13}")
        for label in ["All", "EN", "ES"]:
            if label not in results_by_lang:
                continue
            r = results_by_lang[label]
            icm = r.get(f"{task}/soft/ICMSoft", float("nan"))
            icmn = r.get(f"{task}/soft/ICMSoftNorm", float("nan"))
            ce = r.get(f"{task}/soft/CrossEntropy", float("nan"))
            print(f"  {label:<5} | {icm:>10.4f} | {icmn:>12.4f} | {ce:>13.4f}")

        print(f"  {'Split':<5} | {'ICM':>10} | {'ICMNorm':>12} | {'FMeasure':>13}")
        for label in ["All", "EN", "ES"]:
            if label not in results_by_lang:
                continue
            r = results_by_lang[label]
            icm = r.get(f"{task}/hard/ICM", float("nan"))
            icmn = r.get(f"{task}/hard/ICMNorm", float("nan"))
            fm = r.get(f"{task}/hard/FMeasure", float("nan"))
            print(f"  {label:<5} | {icm:>10.4f} | {icmn:>12.4f} | {fm:>13.4f}")

    out_path = run_dir / "val_metrics_full_langs_v2.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                lang: {k: round(v, 6) for k, v in metrics.items()}
                for lang, metrics in results_by_lang.items()
            },
            f, indent=2,
        )
    print(f"\nSaved -> {out_path}")

    return results_by_lang


def main():
    """
    RUN_DIR = "/data/Medz/exist-models/multitask_ensemble"
    RUN_DIR = "/data/Medz/exist-2026/output/results/task_1/multitask_all"
    RUN_DIR = "/data/Medz/exist-models/multitask_all_aug"
    RUN_DIR = "/data/Medz/exist-models/multitask_ensemble_aug"

    --run-dir=<PATH_TO_TRAINED_MODEL_DIRECTORY>

    --json-path=<PATH_TO>/processed_data.json
    --json-path=<PATH_TO>/processed_data_augmented.json
    """
    DATA = PathManager.DATA_EXIST_DIR

    parser = argparse.ArgumentParser(
        description="Reload a saved checkpoint and recompute full PyEvALL validation metrics broken down by language."
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run directory containing best_model.pt")
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
        "--seed", type=int, default=42, help="Random seed. It must match the original training run (default: 42)"
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
        "--image-model", type=str, default="openai/clip-vit-base-patch32", help="HuggingFace image encoder model name"
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
