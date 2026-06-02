from collections import Counter

import numpy as np
import torch

from exist_2026.consts.task_config import Config, Task22, Task23


def extract_task_2_1_target(item: dict) -> torch.Tensor:
    """Task 2.1 binary yes or no, soft distribution of p(no) and p(yes)."""
    labels = item.get("labels_task2_1", [])
    valid = [l for l in labels if l.upper() not in Config.SKIP_LABELS]
    if valid:
        yes_ratio = np.mean([1 if l.upper() == "YES" else 0 for l in valid])
    else:
        yes_ratio = 0.5
    return torch.tensor([1.0 - yes_ratio, yes_ratio], dtype=torch.float32)


def extract_task_2_2_target(item: dict) -> torch.Tensor:
    """
    Task 2.2 source intention no, direct or judgemental.

    Unknowns are discarded and label "-" is mapped to no
    """
    labels = item.get("labels_task2_2", [])
    counts = Counter()
    total = 0
    for l in labels:
        l_upper = l.upper().strip()
        if l_upper in ("UNKNOWN", ""):
            continue
        if l_upper == "-":
            l_upper = "NO"
        if l_upper in Task22.LABEL_TO_INDEX:
            counts[l_upper] += 1
            total += 1

    dist = torch.zeros(Task22.NUM_CLASSES, dtype=torch.float32)
    if total > 0:
        for label, count in counts.items():
            dist[Task22.LABEL_TO_INDEX[label]] = count / total
    else:
        dist[:] = 1.0 / Task22.NUM_CLASSES
    return dist


def extract_task_2_3_target(item: dict) -> torch.Tensor:
    labels_list = item.get("labels_task2_3", [])
    dist = torch.zeros(Task23.NUM_CLASSES, dtype=torch.float32)
    n_valid = 0

    for annotator_labels in labels_list:
        if isinstance(annotator_labels, str):
            annotator_labels = [annotator_labels]

        valid_labels_for_annotator = []
        for l in annotator_labels:
            l_upper = l.upper().strip()
            if l_upper in ("UNKNOWN", ""):
                continue
            if l_upper == "-":
                l_upper = "NO"
            if l_upper in Task23.LABEL_TO_INDEX:
                valid_labels_for_annotator.append(l_upper)

        if not valid_labels_for_annotator:
            continue

        n_valid += 1
        for valid_label in valid_labels_for_annotator:
            dist[Task23.LABEL_TO_INDEX[valid_label]] += 1

    if n_valid > 0:
        dist /= n_valid
    else:
        dist[0] = 1.0
    return dist
