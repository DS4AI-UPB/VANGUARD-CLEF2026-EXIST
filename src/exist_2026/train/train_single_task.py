import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from tqdm import tqdm

from exist_2026.evaluate.eval_multitask import (
    evaluate_epoch_2_1,
    evaluate_epoch_2_2,
    evaluate_epoch_2_3,
    save_submission_2_1,
    save_submission_2_2,
    save_submission_2_3,
)
from exist_2026.path_manager import PathManager
from exist_2026.train.helpers import pretrain_sensor_autoencoder, build_multitask_model, \
    build_optimizer, find_optimal_threshold, build_multitask_dataloaders, get_raw_data, compute_true_2_1_ratios
from exist_2026.train.nn.determinism import seed_everything
from exist_2026.train.nn.early_stop import EarlyStopping
from exist_2026.train.nn.losses import (
    kl_soft_loss,
    multilabel_soft_bce_loss,
    supervised_contrastive_loss,
)
from exist_2026.train.train_steps import train_step
from exist_2026.visualization.plot_training import plot_training_results


def compute_loss_2_1(outputs, batch, aux_weight=0.3, contrast_weight=0.1):
    """KL on soft distribution + aux CE + contrastive."""
    log_probs = outputs["log_probs_2_1"]
    target = batch["target_2_1"]
    p_yes = target[:, 1]

    w = torch.exp(-(p_yes * (1 - p_yes)))
    kl = (F.kl_div(log_probs, target, reduction="none").sum(dim=1) * w).mean()

    aux = F.cross_entropy(outputs["aux_sexism"], target.argmax(dim=1))

    hard = (p_yes >= 0.5).long()
    con = supervised_contrastive_loss(outputs["contrast_feat"], hard)

    total = kl + aux_weight * aux + contrast_weight * con
    return total, {"loss_kl": kl.item(), "loss_aux": aux.item(), "loss_con": con.item(), "loss_total": total.item()}


def compute_loss_2_2(outputs, batch, contrast_weight=0.1):
    """KL on 3-class soft distribution + contrastive."""
    log_probs = outputs["log_probs_2_2"]
    target = batch["target_2_2"]
    kl = kl_soft_loss(log_probs, target)

    hard = target.argmax(dim=1)
    con = supervised_contrastive_loss(outputs["contrast_feat"], hard)

    total = kl + contrast_weight * con
    return total, {"loss_kl": kl.item(), "loss_con": con.item(), "loss_total": total.item()}


def compute_loss_2_3(outputs, batch, contrast_weight=0.1):
    """BCE on 6-class multi-label soft distribution + contrastive."""
    logits = outputs["logits_2_3"]
    target = batch["target_2_3"]
    bce = multilabel_soft_bce_loss(logits, target)

    # Use "NO" class (index 0) as binary label for contrastive
    hard = (target[:, 0] < 0.5).long()  # 1 = sexist, 0 = not sexist
    con = supervised_contrastive_loss(outputs["contrast_feat"], hard)

    total = bce + contrast_weight * con
    return total, {"loss_bce": bce.item(), "loss_con": con.item(), "loss_total": total.item()}


LOSS_FN = {
    "2.1": compute_loss_2_1,
    "2.2": compute_loss_2_2,
    "2.3": compute_loss_2_3,
}


def train_step_single(model, loader, optimizer, device, task, accumulation_steps=4):
    """Per-task training step using LOSS_FN[task]."""
    model.train()
    loss_fn = LOSS_FN[task]
    running: dict[str, float] = {}
    count = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Train {task}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        loss, log = loss_fn(outputs, batch)
        (loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        for k, v in log.items():
            running[k] = running.get(k, 0.0) + v
        count += 1
        pbar.set_postfix({"loss": f"{log['loss_total']:.4f}"})

    return {k: v / count for k, v in running.items()}


@torch.inference_mode()
def val_step(model, loader, device, task):
    model.eval()
    loss_fn = LOSS_FN[task]
    running: dict[str, float] = {}
    count = 0

    all_ids, all_preds = [], []

    for batch in tqdm(loader, desc=f"Val {task}"):
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch_dev)
        _, log = loss_fn(outputs, batch_dev)

        for k, v in log.items():
            running[k] = running.get(k, 0.0) + v
        count += 1

        all_ids.extend(batch["id"])

        if task == "2.1":
            probs = torch.exp(outputs["log_probs_2_1"]).cpu().numpy()
            all_preds.extend(probs[:, 1].tolist())
        elif task == "2.2":
            probs = torch.exp(outputs["log_probs_2_2"]).cpu().numpy()
            all_preds.extend(probs.tolist())
        elif task == "2.3":
            probs = torch.sigmoid(outputs["logits_2_3"]).cpu().numpy()
            all_preds.extend(probs.tolist())

    return {k: v / count for k, v in running.items()}, all_ids, all_preds


def train_single_task(
        task: str,
        json_path: str | Path,
        img_dir: str | Path,
        test_json: str | Path | None,
        test_img_dir: str | Path | None,
        save_dir: str | Path,
        seed: int = 42,
        train_ratio: float = 0.8,
        num_epochs: int = 50,
        text_model: str = "FacebookAI/xlm-roberta-base",
        image_model: str = "openai/clip-vit-base-patch32",
        lora_r: int = 16,
        lora_alpha: int = 32,
):
    assert task in ("2.1", "2.2", "2.3"), f"Unknown task: {task}"
    seed_everything(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"\tTraining SINGLE TASK {task}")
    print(f"\tDevice: {device}")
    print(f"{'=' * 60}\n")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    accumulation_steps = 4
    early_stop_patience = 5

    train_loader, val_loader, test_loader, scaler, _, _ = build_multitask_dataloaders(
        text_model=text_model, image_model=image_model,
        json_path=Path(json_path), img_dir=Path(img_dir),
        test_json=Path(test_json) if test_json else None,
        test_img_dir=Path(test_img_dir) if test_img_dir else None,
        tasks={task}, train_ratio=train_ratio, batch_size=batch_size, seed=seed,
    )

    config = dict(
        task=task, seed=seed, train_ratio=train_ratio, num_epochs=num_epochs,
        text_model=text_model, image_model=image_model,
        lora_r=lora_r, lora_alpha=lora_alpha, batch_size=batch_size,
        accumulation_steps=accumulation_steps, device=str(device),
        num_train=len(train_loader.dataset), num_val=len(val_loader.dataset),
        num_test=len(test_loader.dataset) if test_loader else 0,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    joblib.dump(scaler, save_dir / "sensor_scaler.joblib")

    pretrain_sensor_autoencoder(train_loader, device)

    model = build_multitask_model(text_model, image_model, device, {task}, lora_r, lora_alpha)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    early_stopping = EarlyStopping(save_dir=save_dir, patience=early_stop_patience, verbose=True)

    csv_path = save_dir / "training_log.csv"
    headers = ["Epoch", "Train_Loss", "Val_Loss", "Val_ICM", "Val_ICMSoft", "Threshold", "Time_s"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(headers)

    raw_data = get_raw_data(val_loader)

    best_threshold = 0.5

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch:02d}/{num_epochs:02d} (Task {task}) ---")

        train_logs = train_step_single(model, train_loader, optimizer, device, task, accumulation_steps)
        val_logs, val_ids, val_preds = val_step(model, val_loader, device, task)

        if task == "2.1":
            true_labels = compute_true_2_1_ratios(val_ids, raw_data)
            best_threshold, best_f1 = find_optimal_threshold(true_labels, val_preds)
            official = evaluate_epoch_2_1(val_ids, val_preds, raw_data, threshold=best_threshold)
            print(f"\tThreshold: {best_threshold:.2f} (F1={best_f1:.4f})")

        elif task == "2.2":
            official = evaluate_epoch_2_2(val_ids, val_preds, raw_data)

        elif task == "2.3":
            official = evaluate_epoch_2_3(val_ids, val_preds, raw_data, threshold=best_threshold)

        elapsed = time.time() - t0

        icm_hard = official.get(f"{task}/hard/ICM", 0.0)
        icm_soft = official.get(f"{task}/soft/ICMSoft", 0.0)

        print(f"\tTrain loss: {train_logs['loss_total']:.4f}")
        print(f"\tVal   loss: {val_logs['loss_total']:.4f}")
        print(f"\tICM:     {icm_hard:.4f}")
        print(f"\tICMSoft: {icm_soft:.4f}")
        for k, v in sorted(official.items()):
            if "ICM" not in k:
                print(f"\t{k}: {v:.4f}")
        print(f"\tTime: {elapsed:.1f}s")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_logs["loss_total"], val_logs["loss_total"],
                                    icm_hard, icm_soft, best_threshold, elapsed])

        scheduler.step(val_logs["loss_total"])
        early_stopping(val_logs["loss_total"], model, None)
        if early_stopping.early_stop:
            break

    with open(save_dir / "best_threshold.json", "w") as f:
        json.dump({"task": task, "threshold": best_threshold}, f, indent=2)

    try:
        plot_training_results(csv_path, save_dir=save_dir)
    except Exception as e:
        print(f"Warning: plots failed: {e}")

    print(f"\n--- Loading best model for test (Task {task}) ---")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))

    if test_loader is not None:
        _, test_ids, test_preds = val_step(model, test_loader, device, task)
        sub_dir = save_dir / "exist2026_VANGUARD"

        if task == "2.1":
            save_submission_2_1(test_ids, test_preds, sub_dir, "VANGUARD", run_id=1, threshold=best_threshold)
        elif task == "2.2":
            save_submission_2_2(test_ids, test_preds, sub_dir, "VANGUARD", run_id=1)
        elif task == "2.3":
            save_submission_2_3(test_ids, test_preds, sub_dir, "VANGUARD", run_id=1, threshold=best_threshold)

        test_raw = test_loader.dataset.data if hasattr(test_loader.dataset, "data") else []
        if test_raw:
            if task == "2.1":
                test_metrics = evaluate_epoch_2_1(test_ids, test_preds, test_raw, threshold=best_threshold)
            elif task == "2.2":
                test_metrics = evaluate_epoch_2_2(test_ids, test_preds, test_raw)
            else:
                test_metrics = evaluate_epoch_2_3(test_ids, test_preds, test_raw, threshold=best_threshold)

            print(f"\nFINAL TEST (Task {task}):")
            for k, v in sorted(test_metrics.items()):
                print(f"\t{k}: {v:.4f}")
            with open(save_dir / "test_results.json", "w") as f:
                json.dump({k: round(v, 6) for k, v in test_metrics.items()}, f, indent=2)
    else:
        print("No test set. Finished!")

    print(f"\n\t- Single-task {task} training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXIST 2026 single-task trainer")
    parser.add_argument("--task", type=str, required=True, choices=["2.1", "2.2", "2.3", "all"],
                        help="Which subtask to train (or 'all' for sequential)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_PATH = PathManager.DATA_EXIST_DIR
    tasks_to_run = ["2.1", "2.2", "2.3"] if args.task == "all" else [args.task]

    for t in tasks_to_run:
        run_dir = PathManager.TASK_1_DIR / f"tastsdt_test_single_task_{t.replace('.', '_')}_r{args.lora_r}_a{args.lora_alpha}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_single_task(
            task=t,
            json_path=DATA_PATH / "training" / "processed_data.json",
            img_dir=DATA_PATH / "training" / "memes",
            test_json=DATA_PATH / "test" / "processed_data.json",
            test_img_dir=DATA_PATH / "test" / "memes",
            save_dir=run_dir,
            seed=args.seed,
            num_epochs=args.epochs,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )

    if len(tasks_to_run) > 1:
        print(f"\n{'=' * 60}")
        print("\tSummary across all single-task runs")
        print(f"{'=' * 60}\n")
        for t in tasks_to_run:
            run_dir = PathManager.TASK_1_DIR / f"single_task_{t.replace('.', '_')}_r{args.lora_r}_a{args.lora_alpha}"
            results_path = run_dir / "test_results.json"
            if results_path.exists():
                with open(results_path) as f:
                    res = json.load(f)
                print(f"\tTask {t}:")
                for k, v in sorted(res.items()):
                    print(f"\t\t{k}: {v:.4f}" if isinstance(v, float) else f"\t\t{k}: {v}")
            else:
                print(f"\tTask {t}: no test results found")
