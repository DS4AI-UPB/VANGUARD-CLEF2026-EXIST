import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from exist_2026.train.nn.losses import MultitaskLoss, compute_loss
from exist_2026.train.nn.meme_classifier import LoRAMemeMultitaskModel, LoRAMemeModel


def train_step(
        model: LoRAMemeMultitaskModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: MultitaskLoss,
        device: torch.device,
        accumulation_steps: int = 4,
) -> dict[str, float]:
    model.train()
    running_logs: dict[str, float] = {}
    count = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        loss, log = criterion(outputs, batch)
        (loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        for k, v in log.items():
            running_logs[k] = running_logs.get(k, 0.0) + v
        count += 1
        pbar.set_postfix({"loss": f"{log['loss_total']:.4f}"})

    return {k: v / count for k, v in running_logs.items()}


@torch.inference_mode()
def val_step(
        model: LoRAMemeMultitaskModel,
        loader: DataLoader,
        criterion: MultitaskLoss,
        device: torch.device,
        tasks: set[str],
) -> tuple[dict[str, float], dict[str, list]]:
    """
    :return:
        - avg_logs: dict of averaged loss components
        - predictions: {
            "ids": list[str],
            "probs_2_1": list[float]         (P(YES) per sample),
            "probs_2_2": list[list[float]]    (3-class per sample),
            "probs_2_3": list[list[float]]    (6-class per sample),
        }
    """
    model.eval()
    running_logs: dict[str, float] = {}
    count = 0

    all_ids: list[str] = []
    all_probs_2_1: list[float] = []
    all_probs_2_2: list[list[float]] = []
    all_probs_2_3: list[list[float]] = []

    for batch in tqdm(loader, desc="Validating"):
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch_dev)
        _, log = criterion(outputs, batch_dev)

        for k, v in log.items():
            running_logs[k] = running_logs.get(k, 0.0) + v
        count += 1

        all_ids.extend(batch["id"])

        if "2.1" in tasks and "log_probs_2_1" in outputs:
            probs = torch.exp(outputs["log_probs_2_1"]).cpu().numpy()
            all_probs_2_1.extend(probs[:, 1].tolist())  # P(YES)

        if "2.2" in tasks and "log_probs_2_2" in outputs:
            probs = torch.exp(outputs["log_probs_2_2"]).cpu().numpy()
            all_probs_2_2.extend(probs.tolist())

        if "2.3" in tasks and "logits_2_3" in outputs:
            probs = torch.sigmoid(outputs["logits_2_3"]).cpu().numpy()
            all_probs_2_3.extend(probs.tolist())

    avg_logs = {k: v / count for k, v in running_logs.items()}
    predictions = {
        "ids": all_ids,
        "probs_2_1": all_probs_2_1 if all_probs_2_1 else None,
        "probs_2_2": all_probs_2_2 if all_probs_2_2 else None,
        "probs_2_3": all_probs_2_3 if all_probs_2_3 else None,
    }
    return avg_logs, predictions


def train_step_t1(
        model: LoRAMemeModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
        accumulation_steps: int = 4, aux_weight: float = 0.3, contrast_weight: float = 0.1
) -> tuple[float, list[float], list[float]]:
    """
    :return: Tuple of (avg_loss, predicted_probs, true_labels)
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        loss = compute_loss(outputs, batch, aux_weight=aux_weight, contrast_weight=contrast_weight) / accumulation_steps
        loss.backward()
        loss_item = loss.item()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_item * accumulation_steps

        probs_yes = torch.exp(outputs["log_probs"][:, 1]).detach().cpu().numpy()
        all_preds.extend(probs_yes)
        all_labels.extend(batch["target_2_1"][:, 1].cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_item * accumulation_steps:.4f}"})

    return total_loss / len(loader), all_preds, all_labels


@torch.inference_mode()
def val_step_t1(
        model: LoRAMemeModel, loader: DataLoader, device: torch.device,
        aux_weight: float = 0.3, contrast_weight: float = 0.1
) -> tuple[float, list[float], list[float], dict[str, float]]:
    """
    :return: Tuple of (avg_loss, predicted_probs, true_labels, {meme_id: prob})
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_ids = [], [], []

    for batch in tqdm(loader, desc="Validating"):
        batch_val = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(batch_val)
        loss = compute_loss(outputs, batch_val, aux_weight=aux_weight, contrast_weight=contrast_weight)
        total_loss += loss.item()
        probs = torch.exp(outputs["log_probs"][:, 1]).cpu().numpy()

        all_preds.extend(probs)
        all_labels.extend(batch_val["target_2_1"][:, 1].cpu().numpy())
        all_ids.extend(batch["id"])

    probs_dict = {mid: float(all_preds[i]) for i, mid in enumerate(all_ids)}
    return total_loss / len(loader), all_preds, all_labels, probs_dict
