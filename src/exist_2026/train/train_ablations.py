import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import torch

from exist_2026.evaluate.eval_multitask import evaluate_epoch_all
from exist_2026.path_manager import PathManager
from exist_2026.train.helpers import (
    build_optimizer,
    pretrain_sensor_autoencoder,
    find_optimal_threshold,
    build_multitask_dataloaders,
    get_raw_data,
    compute_true_2_1_ratios,
)
from exist_2026.train.nn.determinism import seed_everything
from exist_2026.train.nn.early_stop import EarlyStopping
from exist_2026.train.nn.losses import MultitaskLoss
from exist_2026.train.nn.meme_classifier import SensorAutoencoder
from exist_2026.train.train_steps import train_step, val_step
from exist_2026.train.nn.meme_classifier_ablation import LoRAMemeMultitaskModel

DEFAULTS = dict(
    tasks={"2.1", "2.2", "2.3"},
    use_film=True,
    use_description=True,
    use_image=True,
    weight_2_1=1.0,
    weight_2_2=1.0,
    weight_2_3=1.0,
    weight_aux=0.3,
    weight_contrastive=0.1,
)

CONFIGS = [
    {"name": "baseline"},
    {"name": "no_film", "use_film": False},
    {"name": "no_description", "use_description": False},
    {"name": "no_image", "use_image": False},
    {"name": "text_only", "use_description": False, "use_image": False},
    {"name": "no_contrastive", "weight_contrastive": 0.0},
    {"name": "no_aux_head", "weight_aux": 0.0},
    {"name": "only_2_1", "tasks": {"2.1"}},
    {"name": "only_2_2", "tasks": {"2.2"}},
    {"name": "only_2_3", "tasks": {"2.3"}},
]


def build_multitask_model(
        text_model: str, image_model: str, device: torch.device,
        tasks: set[str], lora_r: int = 16, lora_alpha: int = 32,
        use_film: bool = True, use_description: bool = True, use_image: bool = True,
) -> LoRAMemeMultitaskModel:
    model = LoRAMemeMultitaskModel(
        text_model=text_model, image_model=image_model,
        lora_r=lora_r, lora_alpha=lora_alpha, tasks=tasks,
        use_film=use_film, use_description=use_description, use_image=use_image,
    ).to(device)

    sensor_autoencoder = SensorAutoencoder()
    sensor_autoencoder.load_state_dict(
        torch.load(PathManager.SENSOR_WEIGHTS, map_location=device, weights_only=True)
    )
    model.sensorial_encoder.load_state_dict(sensor_autoencoder.encoder.state_dict())
    print("Successfully injected smart sensor weights into the Meme Model!")
    return model


def filter_preds_by_lang(val_preds: dict, raw_data: list[dict], lang: str | None) -> dict:
    """Filter prediction dict by item language ('en', 'es', or None for all)."""
    if lang is None:
        return val_preds
    id_to_lang = {str(item.get("id_EXIST", "")): item.get("lang") for item in raw_data}
    keep = [id_to_lang.get(str(mid)) == lang for mid in val_preds["ids"]]

    def _f(lst):
        return None if lst is None else [v for v, k in zip(lst, keep) if k]

    return {
        "ids": [v for v, k in zip(val_preds["ids"], keep) if k],
        "probs_2_1": _f(val_preds["probs_2_1"]),
        "probs_2_2": _f(val_preds["probs_2_2"]),
        "probs_2_3": _f(val_preds["probs_2_3"]),
    }


def run_ablation(
        config: dict,
        json_path: Path,
        img_dir: Path,
        save_root: Path,
        seed: int = 42,
        train_ratio: float = 0.8,
        num_epochs: int = 30,
        batch_size: int = 16,
        accumulation_steps: int = 4,
        early_patience: int = 4,
        text_model: str = "FacebookAI/xlm-roberta-base",
        image_model: str = "openai/clip-vit-base-patch32",
        lora_r: int = 16,
        lora_alpha: int = 32,
) -> None:
    """Train a single ablation configuration and dump val metrics to disk."""

    cfg = {**DEFAULTS, **config}
    name = cfg["name"]
    save_dir = Path(save_root) / name / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 72)
    print(f"#  ABLATION: {name} (seed={seed})")
    print(f"#  use_film={cfg['use_film']}  use_description={cfg['use_description']}  "
          f"use_image={cfg['use_image']}")
    print(f"#  tasks={sorted(cfg['tasks'])}  w_aux={cfg['weight_aux']}  "
          f"w_con={cfg['weight_contrastive']}")
    print("#" * 72)

    seed_everything(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(save_dir / "ablation_config.json", "w") as f:
        json.dump({k: (sorted(v) if isinstance(v, set) else v) for k, v in cfg.items()},
                  f, indent=2)

    train_loader, val_loader, _, scaler, _, _ = build_multitask_dataloaders(
        text_model=text_model, image_model=image_model,
        json_path=json_path, img_dir=img_dir,
        test_json=None, test_img_dir=None,
        tasks=cfg["tasks"], train_ratio=train_ratio, batch_size=batch_size, seed=seed,
    )
    joblib.dump(scaler, save_dir / "sensor_scaler.joblib")

    pretrain_sensor_autoencoder(train_loader, device)

    model = build_multitask_model(
        text_model=text_model, image_model=image_model, device=device,
        tasks=cfg["tasks"], lora_r=lora_r, lora_alpha=lora_alpha,
        use_film=cfg["use_film"],
        use_description=cfg["use_description"],
        use_image=cfg["use_image"],
    )
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=1)
    early_stopping = EarlyStopping(save_dir=save_dir, patience=early_patience, verbose=True)

    criterion = MultitaskLoss(
        weight_2_1=cfg["weight_2_1"], weight_2_2=cfg["weight_2_2"], weight_2_3=cfg["weight_2_3"],
        weight_aux=cfg["weight_aux"], weight_contrastive=cfg["weight_contrastive"],
        tasks=cfg["tasks"],
    )

    best_thresh_2_1 = 0.5
    raw_data = get_raw_data(val_loader)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        print(f"\n--- {name} (seed={seed}): Epoch {epoch:02d}/{num_epochs:02d} ---")

        train_logs = train_step(model, train_loader, optimizer, criterion, device, accumulation_steps)
        val_logs, val_preds = val_step(model, val_loader, criterion, device, cfg["tasks"])

        if val_preds["probs_2_1"] is not None:
            true_21 = compute_true_2_1_ratios(val_preds["ids"], raw_data)
            best_thresh_2_1, _ = find_optimal_threshold(true_21, val_preds["probs_2_1"])

        print(f"\ttrain_loss={train_logs.get('loss_total', 0):.4f}  "
              f"val_loss={val_logs.get('loss_total', 0):.4f}  "
              f"time={time.time() - t0:.1f}s")

        scheduler.step(val_logs.get("loss_total", 0))
        early_stopping(val_logs.get("loss_total", 0), model, None)
        if early_stopping.early_stop:
            print(f"\tEarly stop triggered at epoch {epoch}")
            break

    # Final evaluation: load best checkpoint and run PyEvALL on All/EN/ES
    print(f"\n--- {name} (seed={seed}): Final evaluation ---")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    _, val_preds = val_step(model, val_loader, criterion, device, cfg["tasks"])

    if val_preds["probs_2_1"] is not None:
        true_21 = compute_true_2_1_ratios(val_preds["ids"], raw_data)
        best_thresh_2_1, _ = find_optimal_threshold(true_21, val_preds["probs_2_1"])

    results_by_lang = {}
    for label, lang in [("All", None), ("EN", "en"), ("ES", "es")]:
        subset = filter_preds_by_lang(val_preds, raw_data, lang)
        if not subset["ids"]:
            continue
        results_by_lang[label] = evaluate_epoch_all(
            ids=subset["ids"],
            probs_2_1=subset["probs_2_1"],
            probs_2_2=subset["probs_2_2"],
            probs_2_3=subset["probs_2_3"],
            dataset_data=raw_data,
            threshold_2_1=best_thresh_2_1,
            threshold_2_3=0.5,
        )

    out_path = save_dir / "val_metrics_full.json"
    with open(out_path, "w") as f:
        json.dump(
            {lang: {k: round(v, 6) for k, v in m.items()} for lang, m in results_by_lang.items()},
            f, indent=2,
        )
    print(f"\tSaved -> {out_path}")

    del model, optimizer, criterion, early_stopping
    torch.cuda.empty_cache()


def main():
    """
        Configurations (10 total):
        1. baseline         -> full DL system, all three tasks, all components on
        2. no_film          -> FiLM modulation off (no human conditioning)
        3. no_description   -> visual description stream off (text + image only)
        4. no_image         -> image stream off (text + description only)
        5. text_only        -> description + image both off
        6. no_contrastive   -> SupCon loss weight set to 0
        7. no_aux_head      -> auxiliary sexism head loss weight set to 0
        8. only_2_1         -> train only on Subtask 2.1
        9. only_2_2         -> train only on Subtask 2.2
        10. only_2_3        -> train only on Subtask 2.3

        Usage:
            # Run all configs sequentially:
            python train_ablations.py

            # (3 seeds on X GPUs in 3 terminals = one terminal per seed):
            CUDA_VISIBLE_DEVICES=1 uv run src/exist_2026/train/train_ablations.py --seed 43
            CUDA_VISIBLE_DEVICES=2 uv run src/exist_2026/train/train_ablations.py --seed 44
            CUDA_VISIBLE_DEVICES=3 uv run src/exist_2026/train/train_ablations.py --seed 45
            CUDA_VISIBLE_DEVICES=0 uv run src/exist_2026/train/train_ablations.py --seed 46

            # Run a subset (for parallel execution across terminals):
            CUDA_VISIBLE_DEVICES=1 uv run src/exist_2026/train/train_ablations.py --names baseline,no_film,no_description,text_only
            CUDA_VISIBLE_DEVICES=2 uv run src/exist_2026/train/train_ablations.py --names no_image,no_contrastive,only_2_1
            CUDA_VISIBLE_DEVICES=3 uv run src/exist_2026/train/train_ablations.py --names no_aux_head,only_2_2,only_2_3

        """
    parser = argparse.ArgumentParser(
        description="Run ablation training configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available config names:\n  " + "\n  ".join(c["name"] for c in CONFIGS),
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated list of config names to run. If omitted, runs all configs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for this run. Results are saved under .../{config}/seed_{seed}/. "
             "Different seeds yield independent train/val splits and model initializations.",
    )
    args = parser.parse_args()
    DATA = PathManager.DATA_EXIST_DIR

    JSON_FILENAME = "processed_data.json"

    JSON_PATH = DATA / "training" / JSON_FILENAME
    IMG_DIR = DATA / "training" / "memes"
    SAVE_ROOT = PathManager.TASK_1_DIR / "ablations"

    SKIP_DONE = True

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.names:
        requested = {n.strip() for n in args.names.split(",") if n.strip()}
        known = {c["name"] for c in CONFIGS}
        unknown = requested - known
        if unknown:
            print(f"[!] Unknown config names: {sorted(unknown)}")
            print(f"    Known: {sorted(known)}")
            sys.exit(1)
        configs_to_run = [c for c in CONFIGS if c["name"] in requested]
        print(f"Running {len(configs_to_run)} configs (seed={args.seed}, filtered by --names): "
              f"{[c['name'] for c in configs_to_run]}")
    else:
        configs_to_run = CONFIGS
        print(f"Running all {len(configs_to_run)} configs (seed={args.seed})")

    for config in configs_to_run:
        result_path = SAVE_ROOT / config["name"] / f"seed_{args.seed}" / "val_metrics_full.json"
        if SKIP_DONE and result_path.exists():
            print(f"[skip] {config['name']}/seed_{args.seed} already has results at {result_path}")
            continue
        try:
            run_ablation(
                config=config,
                json_path=JSON_PATH,
                img_dir=IMG_DIR,
                save_root=SAVE_ROOT,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[!] Ablation {config['name']} (seed={args.seed}) failed: {e}")
            import traceback

            traceback.print_exc()
            print("Continuing with next ablation...")
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print(f"This batch finished (seed={args.seed}). When all seeds are done, run:")
    print("\tpython aggregate_ablations.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
