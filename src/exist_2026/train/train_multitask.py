import json
import json
import time
from pathlib import Path

import joblib
import torch

from exist_2026.evaluate.eval_multitask import (
    evaluate_epoch_all,
    save_submission_2_1,
    save_submission_2_2,
    save_submission_2_3,
)
from exist_2026.train.helpers import pretrain_sensor_autoencoder, build_multitask_model, \
    build_optimizer, find_optimal_threshold, save_config, build_multitask_dataloaders, init_multitask_csv_log, \
    get_raw_data, compute_true_2_1_ratios, log_multitask_epoch, parse_multitask_train_args
from exist_2026.train.nn.determinism import seed_everything
from exist_2026.train.nn.early_stop import EarlyStopping
from exist_2026.train.nn.losses import MultitaskLoss
from exist_2026.train.train_steps import train_step, val_step
from exist_2026.visualization.plot_training import plot_training_results


def train_multitask(
        json_path: str | Path,
        img_dir: str | Path,
        test_json: str | Path | None,
        test_img_dir: str | Path | None,
        save_dir: str | Path,
        tasks: set[str] | None = None,
        seed: int = 42,
        train_ratio: float = 0.8,
        num_epochs: int = 50,
        text_model: str = "FacebookAI/xlm-roberta-base",
        image_model: str = "openai/clip-vit-base-patch32",
        lora_r: int = 16,
        lora_alpha: int = 32,
        weight_2_1: float = 1.0,
        weight_2_2: float = 1.0,
        weight_2_3: float = 1.0,
        weight_aux: float = 0.3,
        weight_contrastive: float = 0.1,
) -> None:
    tasks = tasks or {"2.1", "2.2", "2.3"}
    seed_everything(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Tasks: {sorted(tasks)}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    accumulation_steps = 4
    scheduler_factor = 0.5
    scheduler_patience = 1
    early_stop_patience = 5
    ae_epochs = 50
    ae_lr = 1e-3

    train_loader, val_loader, test_loader, scaler, _, _ = build_multitask_dataloaders(
        text_model=text_model, image_model=image_model,
        json_path=Path(json_path), img_dir=Path(img_dir),
        test_json=Path(test_json) if test_json else None,
        test_img_dir=Path(test_img_dir) if test_img_dir else None,
        tasks=tasks, train_ratio=train_ratio, batch_size=batch_size, seed=seed,
    )

    save_config(
        save_dir, seed=seed, tasks=tasks, train_ratio=train_ratio,
        num_epochs=num_epochs, text_model=text_model, image_model=image_model,
        lora_r=lora_r, lora_alpha=lora_alpha, batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        weight_2_1=weight_2_1, weight_2_2=weight_2_2, weight_2_3=weight_2_3,
        weight_aux=weight_aux, weight_contrastive=weight_contrastive,
        device=device,
        num_train=len(train_loader.dataset), num_val=len(val_loader.dataset),
        num_test=len(test_loader.dataset) if test_loader else 0,
    )
    joblib.dump(scaler, save_dir / "sensor_scaler.joblib")

    pretrain_sensor_autoencoder(train_loader, device, epochs=ae_epochs, learning_rate=ae_lr)

    model = build_multitask_model(text_model, image_model, device, tasks, lora_r, lora_alpha)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )
    early_stopping = EarlyStopping(save_dir=save_dir, patience=early_stop_patience, verbose=True)
    criterion = MultitaskLoss(
        weight_2_1=weight_2_1, weight_2_2=weight_2_2, weight_2_3=weight_2_3,
        weight_aux=weight_aux, weight_contrastive=weight_contrastive, tasks=tasks,
    )

    csv_path = save_dir / "training_log.csv"
    init_multitask_csv_log(csv_path, tasks)
    raw_data = get_raw_data(val_loader)

    best_threshold_2_1 = 0.5
    best_threshold_2_3 = 0.5

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        print(f"\n{'─' * 60}\n  Epoch {epoch:02d}/{num_epochs:02d}\n{'─' * 60}")

        train_logs = train_step(model, train_loader, optimizer, criterion, device, accumulation_steps)
        val_logs, val_preds = val_step(model, val_loader, criterion, device, tasks)

        official = evaluate_epoch_all(
            ids=val_preds["ids"],
            probs_2_1=val_preds["probs_2_1"],
            probs_2_2=val_preds["probs_2_2"],
            probs_2_3=val_preds["probs_2_3"],
            dataset_data=raw_data,
            threshold_2_1=best_threshold_2_1,
            threshold_2_3=best_threshold_2_3,
        )

        if val_preds["probs_2_1"] is not None:
            true_labels_2_1 = compute_true_2_1_ratios(val_preds["ids"], raw_data)
            best_threshold_2_1, best_f1_21 = find_optimal_threshold(true_labels_2_1, val_preds["probs_2_1"])
            print(f"  Task 2.1 threshold: {best_threshold_2_1:.2f} (F1={best_f1_21:.4f})")

        elapsed = time.time() - t0

        print(f"\tTrain loss: {train_logs.get('loss_total', 0):.4f}")
        print(f"\tVal   loss: {val_logs.get('loss_total', 0):.4f}")
        for k, v in sorted(official.items()):
            print(f"\t{k}: {v:.4f}")
        print(f"\tTime: {elapsed:.1f}s")

        log_multitask_epoch(
            csv_path, epoch,
            train_logs.get("loss_total", 0), val_logs.get("loss_total", 0),
            official, tasks, elapsed,
        )

        scheduler.step(val_logs.get("loss_total", 0))
        early_stopping(val_logs.get("loss_total", 0), model, None)
        if early_stopping.early_stop:
            break

    thresholds = {"threshold_2_1": best_threshold_2_1, "threshold_2_3": best_threshold_2_3}
    with open(save_dir / "best_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    try:
        plot_training_results(csv_path, save_dir=save_dir)
    except Exception as e:
        print(f"Warning: could not save plots: {e}")

    print("\n--- Loading best model for final test ---")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))

    if test_loader is not None:
        _, test_preds = val_step(model, test_loader, criterion, device, tasks)

        test_ids = test_preds["ids"]
        sub_dir = save_dir / "exist2026_VANGUARD"

        if test_preds["probs_2_1"] is not None:
            save_submission_2_1(
                test_ids, test_preds["probs_2_1"], sub_dir,
                team_name="VANGUARD", run_id=1, threshold=best_threshold_2_1,
            )

        if test_preds["probs_2_2"] is not None:
            save_submission_2_2(
                test_ids, test_preds["probs_2_2"], sub_dir,
                team_name="VANGUARD", run_id=1,
            )

        if test_preds["probs_2_3"] is not None:
            save_submission_2_3(
                test_ids, test_preds["probs_2_3"], sub_dir,
                team_name="VANGUARD", run_id=1, threshold=best_threshold_2_3,
            )

        # Official test metrics (if gold labels available in training data)
        if hasattr(test_loader.dataset, "data"):
            test_raw = test_loader.dataset.data
        else:
            test_raw = []

        if test_raw:
            test_official = evaluate_epoch_all(
                ids=test_ids,
                probs_2_1=test_preds["probs_2_1"],
                probs_2_2=test_preds["probs_2_2"],
                probs_2_3=test_preds["probs_2_3"],
                dataset_data=test_raw,
                threshold_2_1=best_threshold_2_1,
                threshold_2_3=best_threshold_2_3,
            )
            print("\nFINAL TEST METRICS:")
            for k, v in sorted(test_official.items()):
                print(f"\t{k}: {v:.4f}")

            with open(save_dir / "test_results.json", "w") as f:
                json.dump({k: round(v, 6) for k, v in test_official.items()}, f, indent=2)
    else:
        print("No test set provided. Finished!")

    print("\n Multitask training complete.")


def main():
    args = parse_multitask_train_args()

    train_multitask(
        json_path=args.json_path,
        img_dir=args.img_dir,
        test_json=args.test_json,
        test_img_dir=args.test_img_dir,
        save_dir=args.save_dir,
        tasks=set(args.tasks),
        seed=args.seed,
        train_ratio=args.train_ratio,
        num_epochs=args.num_epochs,
        text_model=args.text_model,
        image_model=args.image_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        weight_2_1=args.weight_2_1,
        weight_2_2=args.weight_2_2,
        weight_2_3=args.weight_2_3,
        weight_aux=args.weight_aux,
        weight_contrastive=args.weight_contrastive,
    )


if __name__ == "__main__":
    # DATA_PATH = PathManager.DATA_EXIST_DIR
    #
    # train_multitask(
    #     json_path=DATA_PATH / "training" / "processed_data.json",
    #     img_dir=DATA_PATH / "training" / "memes",
    #     test_json=DATA_PATH / "test" / "processed_data.json",
    #     test_img_dir=DATA_PATH / "test" / "memes",
    #     save_dir=PathManager.TASK_1_DIR / "tesxt",
    #     tasks={"2.1", "2.2", "2.3"},
    # )
    main()
