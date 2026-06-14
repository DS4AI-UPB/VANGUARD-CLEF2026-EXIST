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
from exist_2026.train.ensemble_pipeline import (
    run_ensemble_for_task,
    apply_ensemble_to_test,
)
from exist_2026.train.helpers import build_multitask_model, build_optimizer, pretrain_sensor_autoencoder, \
    find_optimal_threshold, save_config, build_multitask_dataloaders, get_raw_data, \
    init_multitask_csv_log, compute_true_2_1_ratios, log_multitask_epoch, parse_multitask_train_args
from exist_2026.train.nn.determinism import seed_everything
from exist_2026.train.nn.early_stop import EarlyStopping
from exist_2026.train.nn.losses import MultitaskLoss
from exist_2026.train.train_steps import train_step, val_step
from exist_2026.visualization.plot_training import plot_training_results


def train_multitask_ensemble(
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
):
    tasks = tasks or {"2.1", "2.2", "2.3"}
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Tasks: {sorted(tasks)}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    accum = 4
    early_patience = 5

    train_loader, val_loader, test_loader, scaler, train_idx, val_idx = build_multitask_dataloaders(
        text_model=text_model, image_model=image_model,
        json_path=Path(json_path), img_dir=Path(img_dir),
        test_json=Path(test_json) if test_json else None,
        test_img_dir=Path(test_img_dir) if test_img_dir else None,
        tasks=tasks, train_ratio=train_ratio, batch_size=batch_size, seed=seed,
    )

    save_config(save_dir, seed=seed, tasks=tasks, num_epochs=num_epochs,
                text_model=text_model, image_model=image_model,
                lora_r=lora_r, lora_alpha=lora_alpha, device=device)
    joblib.dump(scaler, save_dir / "sensor_scaler.joblib")

    full_data = get_raw_data(val_loader)
    train_data = [full_data[i] for i in train_idx]
    val_data = [full_data[i] for i in val_idx]

    pretrain_sensor_autoencoder(train_loader, device)

    model = build_multitask_model(text_model, image_model, device, tasks, lora_r, lora_alpha)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    early_stopping = EarlyStopping(save_dir=save_dir, patience=early_patience, verbose=True)
    criterion = MultitaskLoss(
        weight_2_1=weight_2_1, weight_2_2=weight_2_2, weight_2_3=weight_2_3,
        weight_aux=weight_aux, weight_contrastive=weight_contrastive, tasks=tasks,
    )

    csv_path = save_dir / "training_log.csv"
    init_multitask_csv_log(csv_path, tasks)

    best_thresh_2_1 = 0.5

    print("\n" + "=" * 60)
    print("\tPHASE 1: Deep Learning Training")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch:02d}/{num_epochs} ---")

        train_logs = train_step(model, train_loader, optimizer, criterion, device, accum)
        val_logs, val_preds = val_step(model, val_loader, criterion, device, tasks)

        official = evaluate_epoch_all(
            ids=val_preds["ids"],
            probs_2_1=val_preds["probs_2_1"],
            probs_2_2=val_preds["probs_2_2"],
            probs_2_3=val_preds["probs_2_3"],
            dataset_data=full_data,
            threshold_2_1=best_thresh_2_1,
        )

        if val_preds["probs_2_1"] is not None:
            true_21 = compute_true_2_1_ratios(val_preds["ids"], full_data)
            best_thresh_2_1, _ = find_optimal_threshold(true_21, val_preds["probs_2_1"])

        elapsed = time.time() - t0
        print(f"\tTrain loss: {train_logs.get('loss_total', 0):.4f} | Val loss: {val_logs.get('loss_total', 0):.4f}")
        for k, v in sorted(official.items()):
            print(f"\t{k}: {v:.4f}")

        log_multitask_epoch(
            csv_path, epoch,
            train_logs.get("loss_total", 0), val_logs.get("loss_total", 0),
            official, tasks, elapsed,
        )

        scheduler.step(val_logs.get("loss_total", 0))
        early_stopping(val_logs.get("loss_total", 0), model, None)
        if early_stopping.early_stop:
            break

    try:
        plot_training_results(csv_path, save_dir=save_dir)
    except Exception as e:
        print(f"Warning: plots failed: {e}")

    print("\n" + "=" * 60)
    print("\tPHASE 2: SVM Ensemble")
    print("=" * 60)

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))

    _, val_preds = val_step(model, val_loader, criterion, device, tasks)
    val_ids = val_preds["ids"]

    ensembles = {}
    dl_val_map = {
        "2.1": val_preds["probs_2_1"],
        "2.2": val_preds["probs_2_2"],
        "2.3": val_preds["probs_2_3"],
    }

    for t in sorted(tasks):
        dl_probs = dl_val_map.get(t)
        if dl_probs is None:
            continue
        ens = run_ensemble_for_task(
            task=t, train_data=train_data, val_data=val_data,
            val_ids=val_ids, dl_val_probs=dl_probs,
            save_dir=save_dir / "ensemble",
        )
        ensembles[t] = ens

    if test_loader is not None:
        print("\n" + "=" * 60)
        print("\tPHASE 3: Test Predictions & Submissions")
        print("=" * 60)

        _, test_preds = val_step(model, test_loader, criterion, device, tasks)
        test_ids = test_preds["ids"]
        test_raw = test_loader.dataset.data if hasattr(test_loader.dataset, "data") else []

        sub_dir = save_dir / "exist2026_VANGUARD"
        dl_test_map = {
            "2.1": test_preds["probs_2_1"],
            "2.2": test_preds["probs_2_2"],
            "2.3": test_preds["probs_2_3"],
        }

        for t in sorted(tasks):
            dl_probs = dl_test_map.get(t)
            if dl_probs is None:
                continue

            if t in ensembles:
                blended = apply_ensemble_to_test(t, ensembles[t], test_raw, test_ids, dl_probs)
                thresh = ensembles[t].threshold
                print(f"\tTask {t}: applied ensemble (DL={ensembles[t].dl_weight:.0%})")
            else:
                blended = dl_probs
                thresh = 0.5

            if t == "2.1":
                save_submission_2_1(test_ids, blended, sub_dir, "VANGUARD", run_id=1, threshold=thresh)
            elif t == "2.2":
                save_submission_2_2(test_ids, blended, sub_dir, "VANGUARD", run_id=1)
            elif t == "2.3":
                save_submission_2_3(test_ids, blended, sub_dir, "VANGUARD", run_id=1, threshold=thresh)

        pure_dir = save_dir / "exist2026_VANGUARD_pure_dl"
        if test_preds["probs_2_1"] is not None:
            save_submission_2_1(test_ids, test_preds["probs_2_1"], pure_dir, "VANGUARD", run_id=2,
                                threshold=best_thresh_2_1)
        if test_preds["probs_2_2"] is not None:
            save_submission_2_2(test_ids, test_preds["probs_2_2"], pure_dir, "VANGUARD", run_id=2)
        if test_preds["probs_2_3"] is not None:
            save_submission_2_3(test_ids, test_preds["probs_2_3"], pure_dir, "VANGUARD", run_id=2, threshold=0.5)

    print("\n- Multitask + Ensemble training complete.")


def main():
    args = parse_multitask_train_args()

    train_multitask_ensemble(
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
    # DATA = PathManager.DATA_EXIST_DIR
    # train_multitask_ensemble(
    #     json_path=DATA / "training" / "processed_data.json",
    #     img_dir=DATA / "training" / "memes",
    #     test_json=DATA / "test" / "processed_data.json",
    #     test_img_dir=DATA / "test" / "memes",
    #     save_dir=PathManager.TASK_1_DIR / "testx",
    #     tasks={"2.1", "2.2", "2.3"},
    # )
    main()
