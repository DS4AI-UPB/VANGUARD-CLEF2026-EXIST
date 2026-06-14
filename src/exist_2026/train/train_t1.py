import json
import time
from pathlib import Path

import torch

from exist_2026.evaluate.eval import evaluate_epoch, save_submission
from exist_2026.path_manager import PathManager
from exist_2026.train.helpers import pretrain_sensor_autoencoder, build_model, build_optimizer, \
    save_config, save_scaler, init_csv_log, compute_metrics, find_optimal_threshold, log_epoch, save_threshold, \
    save_val_probabilities, save_test_results, build_multitask_dataloaders, get_raw_data
from exist_2026.train.nn.determinism import seed_everything
from exist_2026.train.nn.early_stop import EarlyStopping
from exist_2026.train.train_steps import train_step_t1, val_step_t1
from exist_2026.visualization.plot_training import plot_training_results


def train_and_validate_model(
        json_path: str | Path,
        img_dir: str | Path,
        test_json: str | Path,
        test_img_dir: str | Path,
        save_dir: str | Path,
        seed: int = 42,
        train_ratio: float = 0.8,
        num_epochs: int = 50,
        text_model: str = "FacebookAI/xlm-roberta-base",
        image_model: str = "openai/clip-vit-base-patch32",
        lora_r: int = 16,
        lora_alpha: int = 32,
) -> None:
    seed_everything(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    accumulation_steps = 4
    aux_weight = 0.3
    contrast_weight = 0.1
    base_lr = 3e-5
    lora_lr = 8e-6
    weight_decay = 0.1
    scheduler_factor = 0.5
    scheduler_patience = 1
    early_stop_patience = 5
    sensor_autoencoder_epochs = 50
    sensor_autoencoder_lr = 1e-3
    grad_clip = 1.0

    train_loader, val_loader, test_loader, scaler, _, _ = build_multitask_dataloaders(
        text_model=text_model,
        image_model=image_model,
        json_path=Path(json_path),
        img_dir=Path(img_dir),
        test_json=Path(test_json) if test_json else None,
        test_img_dir=Path(test_img_dir) if test_img_dir else None,
        tasks=None,
        train_ratio=train_ratio,
        batch_size=batch_size,
        seed=seed
    )
    save_config(
        save_dir,
        seed=seed,
        train_ratio=train_ratio,
        num_epochs=num_epochs,
        text_model=text_model,
        image_model=image_model,
        json_path=json_path,
        img_dir=img_dir,
        test_json=test_json,
        test_img_dir=test_img_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        aux_weight=aux_weight,
        contrast_weight=contrast_weight,
        base_lr=base_lr,
        lora_lr=lora_lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        weight_decay=weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        early_stop_patience=early_stop_patience,
        ae_epochs=sensor_autoencoder_epochs,
        ae_lr=sensor_autoencoder_lr,
        grad_clip=grad_clip,
        device=device,
        num_train_samples=len(train_loader.dataset),
        num_val_samples=len(val_loader.dataset),
        num_test_samples=len(test_loader.dataset) if test_loader else 0,
    )
    save_scaler(save_dir, scaler)
    print("Fitting Scaler on training sensors...")
    pretrain_sensor_autoencoder(
        train_loader, device=device, epochs=sensor_autoencoder_epochs, learning_rate=sensor_autoencoder_lr
    )

    model = build_model(
        text_model_name=text_model, image_model_name=image_model, device=device, lora_r=lora_r, lora_alpha=lora_alpha
    )
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )
    early_stopping = EarlyStopping(save_dir=save_dir, patience=early_stop_patience, verbose=True)

    csv_file = save_dir / "training_log.csv"
    headers = [
        "Epoch", "Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc", "Train_F1_Macro", "Val_F1_Macro",
        "Train_F1_YES", "Val_F1_YES", "Train_AUC", "Val_AUC", "Threshold", "Time_Seconds"
    ]
    init_csv_log(csv_path=csv_file, headers=headers)
    best_thresh_global = 0.5

    raw_data = get_raw_data(val_loader)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        print(f"\n--- Epoch {epoch:02d}/{num_epochs:02d} ---")

        train_loss, train_preds, train_labels = train_step_t1(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            accumulation_steps=accumulation_steps, aux_weight=aux_weight, contrast_weight=contrast_weight
        )
        train_metrics = compute_metrics(train_preds, train_labels, threshold=0.5)

        val_loss, val_preds, val_labels, val_probs_dict = val_step_t1(
            model=model, loader=val_loader, device=device, aux_weight=aux_weight, contrast_weight=contrast_weight
        )

        val_ids = list(val_probs_dict.keys())
        val_probs = list(val_probs_dict.values())
        best_threshold, best_f1 = find_optimal_threshold(val_labels, val_preds)
        best_thresh_global = best_threshold
        val_metrics = compute_metrics(val_preds, val_labels, threshold=best_thresh_global)
        official_metrics = evaluate_epoch(val_ids, val_probs, raw_data, threshold=best_threshold)

        epoch_time = time.time() - start_time
        print(
            f"\tTrain -> Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f} "
            f"| F1-Macro: {train_metrics['f1_macro']:.4f} | F1-YES: {train_metrics['f1_yes']:.4f} "
            f"| AUC: {train_metrics['auc']:.4f}")
        print(
            f"\tVal   -> Loss: {val_loss:.4f} | Acc: {val_metrics['acc']:.4f} "
            f"| F1-Macro: {val_metrics['f1_macro']:.4f} | F1-YES: {val_metrics['f1_yes']:.4f} "
            f"| AUC: {val_metrics['auc']:.4f}")
        print(f"\tBest Threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
        print(f"\tOfficial ICM:      {official_metrics['hard/ICM']:.4f}")
        print(f"\tOfficial ICMNorm:  {official_metrics['hard/ICMNorm']:.4f}")
        print(f"\tOfficial ICMSoft:  {official_metrics['soft/ICMSoft']:.4f}")
        print(f"\tTime  -> {epoch_time:.1f}s")

        log_epoch(csv_file, epoch, train_loss, val_loss, train_metrics, val_metrics, best_threshold, epoch_time)
        scheduler.step(val_loss)
        early_stopping(val_loss, model, val_probs_dict)
        if early_stopping.early_stop:
            break

    save_threshold(save_dir, best_thresh_global)
    save_val_probabilities(save_dir, val_probs_dict)
    print("\nGenerating training plots...")
    try:
        plot_training_results(csv_file, save_dir=save_dir)
    except Exception as e:
        print(f"Warning: could not save plots: {e}")
    print("\n--- Training Complete. Loading Best Model for Final Test ---")

    if test_loader is None:
        print("Nothing to test. Finished!")
        return

    model.load_state_dict(torch.load(Path(save_dir) / "best_model.pt"))
    _, test_preds, test_labels, test_probs_dict = val_step_t1(model=model, loader=test_loader, device=device)
    test_ids = list(test_probs_dict.keys())
    test_probs = list(test_probs_dict.values())

    test_metrics = compute_metrics(test_preds, test_labels, threshold=best_thresh_global)

    print(f"\nFINAL TEST RESULTS (Threshold: {best_thresh_global:.2f}):")
    print(f"\tTest Accuracy: {test_metrics['acc']:.4f}")
    print(f"\tTest F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"\tTest F1 (YES):   {test_metrics['f1_yes']:.4f}")
    print(f"\tTest AUC:        {test_metrics['auc']:.4f}")
    save_test_results(save_dir, test_metrics, best_thresh_global, test_preds, test_labels)
    save_submission(
        test_ids, test_probs,
        save_dir=save_dir / "exist2026_VANGUARD",
        team_name="VANGUARD", run_id=1, threshold=best_thresh_global
    )


if __name__ == "__main__":
    lora_configs = [
        # (8, 16),
        (16, 32),
        # (32, 64),
    ]
    for r, alpha in lora_configs:
        print(f"\n{'=' * 60}")
        print(f"\tLoRA sweep: r={r}, alpha={alpha}")
        print(f"{'=' * 60}\n")
        run_dir = PathManager.TASK_1_DIR / f"lora_r{r}_a{alpha}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_and_validate_model(
            json_path=PathManager.DATA_DIR / "processed_data.json",
            img_dir=PathManager.DATA_DIR / "memes",
            test_json=PathManager.DATA_DIR / "test" / "processed_data.json",
            test_img_dir=PathManager.DATA_DIR / "test" / "memes",
            save_dir=run_dir,
            lora_r=r,
            lora_alpha=alpha
        )

    print(f"\n{'=' * 60}")
    print("\tLoRA Sweep Summary")
    print(f"{'=' * 60}\n")

    summary = []
    for r, alpha in lora_configs:
        run_dir = PathManager.TASK_1_DIR / f"lora_r{r}_a{alpha}"
        entry = {"r": r, "alpha": alpha}

        csv_path = run_dir / "training_log.csv"
        if csv_path.exists():
            import pandas as pd

            df = pd.read_csv(csv_path)
            best_idx = df["Val_F1_Macro"].idxmax()
            entry["val_f1"] = df.loc[best_idx, "Val_F1_Macro"]
            entry["val_acc"] = df.loc[best_idx, "Val_Acc"]
            entry["val_loss"] = df.loc[best_idx, "Val_Loss"]
            entry["best_epoch"] = int(df.loc[best_idx, "Epoch"])
            entry["total_epochs"] = len(df)

        results_path = run_dir / "test_results.json"
        if results_path.exists():
            with open(results_path) as f:
                res = json.load(f)
            m = res["metrics"]
            entry["test_f1"] = m["f1_macro"]
            entry["test_acc"] = m["acc"]
            entry["test_auc"] = m["auc"]

        summary.append(entry)

    print(
        f"\t{'Config':<12} | {'Val F1':>8} {'Val Acc':>8} {'Val Loss':>9} {'Ep':>3} | {'Test F1':>8} {'Test Acc':>8} {'Test AUC':>9}")
    print(f"\t{'-' * 12}-+-{'-' * 8}-{'-' * 8}-{'-' * 9}-{'-' * 3}-+-{'-' * 8}-{'-' * 8}-{'-' * 9}")

    for e in summary:
        val_str = f"{e.get('val_f1', 0):.4f}   {e.get('val_acc', 0):.4f}   {e.get('val_loss', 0):.5f} {e.get('best_epoch', '-'):>3}"
        test_str = f"{e.get('test_f1', 0):.4f}   {e.get('test_acc', 0):.4f}   {e.get('test_auc', 0):.4f}" if "test_f1" in e else "\tno test results"
        print(f"\tr={e['r']:<2} α={e['alpha']:<4} | {val_str} | {test_str}")

    best_val = max(summary, key=lambda x: x.get("val_f1", 0))
    best_test = max(summary, key=lambda x: x.get("test_f1", 0))

    print(f"\n  Conclusion:")
    print(
        f"\t\tBest validation F1:  r={best_val['r']}, α={best_val['alpha']} ({best_val.get('val_f1', 0):.4f} at epoch {best_val.get('best_epoch', '?')})")
    if best_test.get("test_f1"):
        print(f"\t\tBest test F1:        r={best_test['r']}, α={best_test['alpha']} ({best_test['test_f1']:.4f})")
        if best_val["r"] != best_test["r"]:
            print(f"\t\tWarn: Val and test disagree - check for overfitting on r={best_test['r']}")
        else:
            print(f"\t\tDone: Val and test agree — r={best_val['r']} is the pick")
