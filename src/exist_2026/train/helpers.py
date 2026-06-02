import csv
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from exist_2026.dataset.data_loader import ExistMemeDataset
from exist_2026.path_manager import PathManager
from exist_2026.train.nn.meme_classifier import SensorAutoencoder, LoRAMemeMultitaskModel, LoRAMemeModel


def pretrain_sensor_autoencoder(
        train_loader: DataLoader, device: torch.device, epochs: int = 50, learning_rate: float = 1e-3
) -> None:
    if PathManager.SENSOR_WEIGHTS.exists():
        print("Sensor autoencoder weights found, skipping pretraining...")
        return

    print("\n--- Pre-training Sensor Autoencoder ---")
    start_time_total = time.time()

    autoencoder = SensorAutoencoder().to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    autoencoder.train()
    pbar = tqdm(range(epochs), desc="AE Pre-training")
    for _ in pbar:
        epoch_start_time = time.time()
        total_loss = 0.0

        for batch in train_loader:
            sensors = batch["sensorial"].to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(sensors)
            loss = criterion(reconstructed, sensors)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        pbar.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "Time/Ep": f"{epoch_time:.2f}s"
        })

    total_time = time.time() - start_time_total
    torch.save(autoencoder.state_dict(), PathManager.SENSOR_WEIGHTS)
    print(f"--- Autoencoder Pre-training Complete in {total_time:.1f} seconds! ---\n")


def create_split_indices(data: list[dict], seed: int = 42, train_ratio: float = 0.8) -> tuple[list[int], list[int]]:
    """Ensures augmented copies of the same meme stay in the same split."""
    original_ids = sorted(set(
        item.get("id_EXIST", "").replace("_aug", "") for item in data
    ))
    rng = random.Random(seed)
    rng.shuffle(original_ids)
    train_size = int(train_ratio * len(original_ids))
    train_original_ids = set(original_ids[:train_size])
    val_original_ids = set(original_ids[train_size:])

    train_indices = [
        i for i, item in enumerate(data) if item.get("id_EXIST", "").replace("_aug", "") in train_original_ids
    ]
    val_indices = [
        i for i, item in enumerate(data) if item.get("id_EXIST", "").replace("_aug", "") in val_original_ids
    ]
    return train_indices, val_indices


def build_multitask_dataloaders(
        text_model: str,
        image_model: str,
        json_path: Path,
        img_dir: Path,
        test_json: Path | None,
        test_img_dir: Path | None,
        tasks: set[str] | None,
        train_ratio: float = 0.8,
        batch_size: int = 16,
        seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader | None, StandardScaler, list[int], list[int]]:
    """Shared dataloader builder for the multitask training scripts.

    Returns (train_loader, val_loader, test_loader, scaler, train_idx, val_idx).
    """
    temp_ds = ExistMemeDataset(
        json_path, img_dir, text_model, image_model, is_train=False, scaler=None, tasks=tasks
    )
    train_idx, val_idx = create_split_indices(temp_ds.data, seed=seed, train_ratio=train_ratio)
    print(f"Split: {len(train_idx)} train, {len(val_idx)} validation")

    train_sensors = [temp_ds.data[i].get("processed_sensors", temp_ds.default_sensors) for i in train_idx]
    scaler = StandardScaler()
    scaler.fit(train_sensors)
    temp_ds.scaler = scaler

    train_ds = ExistMemeDataset(
        json_path, img_dir, text_model, image_model, is_train=True, scaler=scaler, tasks=tasks
    )
    train_ds.data = temp_ds.data
    train_ds.tokenizer = temp_ds.tokenizer
    train_ds.image_processor = temp_ds.image_processor

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(temp_ds, val_idx), batch_size=batch_size, shuffle=False)

    test_loader = None
    if test_json is not None and test_img_dir is not None:
        test_ds = ExistMemeDataset(
            test_json, test_img_dir, text_model, image_model, is_train=False, scaler=scaler, tasks=tasks
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, train_idx, val_idx


def get_raw_data(loader: DataLoader) -> list[dict]:
    """Pull raw .data out of a (possibly Subset-wrapped) DataLoader's dataset."""
    if hasattr(loader.dataset, "dataset"):
        return loader.dataset.dataset.data
    return loader.dataset.data


def compute_true_2_1_ratios(ids: list[str], raw_data: list[dict]) -> list[float]:
    """For each id, the YES-ratio across valid annotators (used to tune the 2.1 threshold)."""
    id_map = {str(d.get("id_EXIST", "")): d for d in raw_data}
    out = []
    for mid in ids:
        item = id_map.get(str(mid))
        if item is None:
            out.append(0.5)
            continue
        labels = [l.upper() for l in item.get("labels_task2_1", []) if l.upper() not in {"UNKNOWN", "", "-"}]
        if not labels:
            out.append(0.5)
        else:
            out.append(sum(1 for l in labels if l == "YES") / len(labels))
    return out


def build_model(
        text_model_name: str, image_model_name: str, device: torch.device, lora_r: int = 16, lora_alpha: int = 32
) -> LoRAMemeModel:
    model = LoRAMemeModel(
        text_model=text_model_name, image_model=image_model_name, lora_r=lora_r, lora_alpha=lora_alpha
    ).to(device)

    sensor_autoencoder = SensorAutoencoder()
    sensor_autoencoder.load_state_dict(torch.load(PathManager.SENSOR_WEIGHTS, map_location=device, weights_only=True))
    model.sensorial_encoder.load_state_dict(sensor_autoencoder.encoder.state_dict())
    print("Successfully injected smart sensor weights into the Meme Model!")

    return model


def build_multitask_model(
        text_model: str, image_model: str, device: torch.device,
        tasks: set[str], lora_r: int = 16, lora_alpha: int = 32,
) -> LoRAMemeMultitaskModel:
    model = LoRAMemeMultitaskModel(
        text_model=text_model, image_model=image_model,
        lora_r=lora_r, lora_alpha=lora_alpha, tasks=tasks,
    ).to(device)

    sensor_autoencoder = SensorAutoencoder()
    sensor_autoencoder.load_state_dict(torch.load(PathManager.SENSOR_WEIGHTS, map_location=device, weights_only=True))
    model.sensorial_encoder.load_state_dict(sensor_autoencoder.encoder.state_dict())
    print("Successfully injected smart sensor weights into the Meme Model!")
    return model


def build_optimizer(model: LoRAMemeModel | LoRAMemeMultitaskModel) -> torch.optim.Optimizer:
    lora_params = []
    lora_params += [p for p in model.text_encoder.parameters() if p.requires_grad]
    lora_params += [p for p in model.image_encoder.parameters() if p.requires_grad]

    base_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "text_encoder" not in n and "image_encoder" not in n
    ]

    return torch.optim.AdamW([
        {"params": base_params, "lr": 3e-5},
        {"params": lora_params, "lr": 8e-6},
    ], weight_decay=0.1)


def find_optimal_threshold(
        labels: list[float], preds: list[float], low: float = 0.30, high: float = 0.71, step: float = 0.02
) -> tuple[float, float]:
    """Grid search for the threshold that maximizes macro F1."""
    bin_labels = [1 if l >= 0.5 else 0 for l in labels]
    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(low, high, step):
        temp_preds_bin = [1 if p >= thresh else 0 for p in preds]
        temp_f1 = f1_score(bin_labels, temp_preds_bin, average="macro")
        if temp_f1 > best_f1:
            best_f1, best_thresh = temp_f1, thresh

    return best_thresh, best_f1


def save_config(save_dir: Path, **kwargs) -> None:
    config = {}
    for k, v in kwargs.items():
        if isinstance(v, (Path, torch.device)):
            config[k] = str(v)
        elif isinstance(v, set):
            config[k] = sorted(v)
        else:
            config[k] = v
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["pytorch_version"] = torch.__version__
    config["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config["cuda_device"] = torch.cuda.get_device_name(0)
    path = save_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config → {path}")


def compute_metrics(preds: list[float], labels: list[float], threshold: float) -> dict[str, float]:
    bin_labels = [1 if l >= 0.5 else 0 for l in labels]
    bin_preds = [1 if p >= threshold else 0 for p in preds]

    try:
        auc = roc_auc_score(bin_labels, preds)
    except ValueError:
        print("AUC can happen if only one class is present in bin_labels")
        auc = 0.0

    return {
        "acc": accuracy_score(bin_labels, bin_preds),
        "f1_macro": f1_score(bin_labels, bin_preds, average="macro"),
        "f1_yes": f1_score(bin_labels, bin_preds, pos_label=1, average="binary"),
        "auc": auc,
    }


def init_csv_log(csv_path: Path, headers: list[str]) -> None:
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def init_multitask_csv_log(csv_path: Path, tasks: set[str]) -> None:
    headers = ["Epoch", "Train_Loss", "Val_Loss"]
    for t in sorted(tasks):
        headers.append(f"Val_ICM_{t}")
        headers.append(f"Val_ICMSoft_{t}")
    headers.append("Time_Seconds")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(headers)


def log_multitask_epoch(
        csv_path: Path,
        epoch: int,
        train_loss: float,
        val_loss: float,
        official: dict[str, float],
        tasks: set[str],
        elapsed: float,
) -> None:
    row = [epoch, train_loss, val_loss]
    for t in sorted(tasks):
        row.append(official.get(f"{t}/hard/ICM", 0.0))
        row.append(official.get(f"{t}/soft/ICMSoft", 0.0))
    row.append(elapsed)
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def log_epoch(
        csv_path: Path,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: dict,
        val_metrics: dict,
        threshold: float,
        elapsed: float
) -> None:
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch, train_loss, val_loss,
            train_metrics["acc"], val_metrics["acc"],
            train_metrics["f1_macro"], val_metrics["f1_macro"],
            train_metrics["f1_yes"], val_metrics["f1_yes"],
            train_metrics["auc"], val_metrics["auc"],
            threshold, elapsed
        ])


def save_test_results(save_dir: Path, metrics: dict, threshold: float, preds: list[float], labels: list[float]) -> None:
    results = {
        "threshold": threshold,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "num_samples": len(labels),
        "positive_rate": round(sum(1 for l in labels if l >= 0.5) / len(labels), 4),
        "mean_predicted_prob": round(float(np.mean(preds)), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = save_dir / "test_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved test results to {path}")


def save_val_probabilities(save_dir: Path, probs_dict: dict[str, float]) -> None:
    path = save_dir / "val_probabilities.json"
    with open(path, "w") as f:
        json.dump(probs_dict, f, indent=2)
    print(f"Saved val probabilities to {path}")


def save_threshold(save_dir: Path, threshold: float) -> None:
    path = save_dir / "best_threshold.json"
    with open(path, "w") as f:
        json.dump({"best_threshold": threshold}, f, indent=2)
    print(f"Saved best threshold to {path}")


def save_scaler(save_dir: Path, scaler: StandardScaler) -> None:
    """Save the fitted StandardScaler for reproducible inference."""
    path = save_dir / "sensor_scaler.joblib"
    joblib.dump(scaler, path)
    print(f"Saved scaler to {path}")
