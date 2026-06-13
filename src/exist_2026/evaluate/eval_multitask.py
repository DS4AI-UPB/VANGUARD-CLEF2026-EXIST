import json
from collections import Counter
from pathlib import Path

import numpy as np
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

from exist_2026.consts.task_config import Config, Task23, Task22
from exist_2026.evaluate.eval_utils import _write_tmp_json, _extract_metrics


def probs_to_pyevall_hard_2_1(
        ids: list[str], probs_yes: list[float], threshold: float = 0.5
) -> list[dict]:
    """Convert P(YES) to hard PyEvALL records for subtask 2.1."""
    return [
        {"test_case": Config.TEST_CASE, "id": str(mid), "value": "YES" if p >= threshold else "NO"}
        for mid, p in zip(ids, probs_yes)
    ]


def probs_to_pyevall_soft_2_1(ids: list[str], probs_yes: list[float]) -> list[dict]:
    """Convert P(YES) to soft PyEvALL records for subtask 2.1."""
    return [
        {
            "test_case": Config.TEST_CASE,
            "id": str(mid),
            "value": {"YES": round(float(p), 10), "NO": round(1.0 - float(p), 10)},
        }
        for mid, p in zip(ids, probs_yes)
    ]


def probs_to_pyevall_hard_2_2(
        ids: list[str], probs: list[list[float]]
) -> list[dict]:
    """
    Convert 3-class probabilities to hard PyEvALL records for subtask 2.2.

    :param ids: List of meme IDs.
    :param probs: List of [P(NO), P(DIRECT), P(JUDGEMENTAL)] arrays.
    """
    records = []
    for mid, p in zip(ids, probs):
        label_idx = int(np.argmax(p))
        label = Task22.LABELS[label_idx]
        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": label})
    return records


def probs_to_pyevall_soft_2_2(ids: list[str], probs: list[list[float]]) -> list[dict]:
    """Convert 3-class probabilities to soft PyEvALL records for subtask 2.2."""
    records = []
    for mid, p in zip(ids, probs):
        value = {Task22.LABELS[i]: round(float(p[i]), 10) for i in range(len(Task22.LABELS))}
        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": value})
    return records


def probs_to_pyevall_hard_2_3(
        ids: list[str], probs: list[list[float]], threshold: float = 0.5
) -> list[dict]:
    """
    Convert 6-class multi-label probabilities to hard PyEvALL records for subtask 2.3.

    Each class is independently thresholded. If no positive class exceeds the
    threshold, the prediction is ["NO"].

    :param ids: List of meme IDs.
    :param probs: List of 6D probability vectors (one per class, via sigmoid).
    """
    records = []
    for mid, p in zip(ids, probs):
        selected = []
        for i, label in enumerate(Task23.LABELS):
            if label == "NO":
                continue  # NO is inferred from absence of positive labels
            if p[i] >= threshold:
                selected.append(label)
        if not selected:
            selected = ["NO"]
        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": selected})
    return records


def probs_to_pyevall_soft_2_3(ids: list[str], probs: list[list[float]]) -> list[dict]:
    """Convert 6-class multi-label probabilities to soft PyEvALL records for subtask 2.3."""
    records = []
    for mid, p in zip(ids, probs):
        value = {Task23.LABELS[i]: round(float(p[i]), 10) for i in range(len(Task23.LABELS))}
        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": value})
    return records


def gold_from_dataset_hard_2_1(
        data: list[dict], ids: list[str], annotator_threshold: int = Config.HARD_THRESHOLD_2_1
) -> list[dict]:
    """Build hard gold for subtask 2.1 (same as original eval.py)."""
    id_to_item = {str(item.get("id_EXIST", "")): item for item in data}
    records = []
    for mid in ids:
        item = id_to_item.get(str(mid))
        if item is None:
            continue
        labels = [l.upper() for l in item.get("labels_task2_1", []) if l.upper() not in Config.SKIP_LABELS]
        if not labels:
            continue
        yes_count = sum(1 for l in labels if l == "YES")
        no_count = sum(1 for l in labels if l == "NO")
        if yes_count > annotator_threshold:
            gold_label = "YES"
        elif no_count > annotator_threshold:
            gold_label = "NO"
        else:
            continue
        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": gold_label})
    return records


def gold_from_dataset_soft_2_1(data: list[dict], ids: list[str]) -> list[dict]:
    """Build soft gold for subtask 2.1."""
    id_to_item = {str(item.get("id_EXIST", "")): item for item in data}
    records = []
    for mid in ids:
        item = id_to_item.get(str(mid))
        if item is None:
            continue
        labels = [l.upper() for l in item.get("labels_task2_1", []) if l.upper() not in Config.SKIP_LABELS]
        if not labels:
            continue
        yes_ratio = sum(1 for l in labels if l == "YES") / len(labels)
        records.append({
            "test_case": Config.TEST_CASE,
            "id": str(mid),
            "value": {"YES": round(yes_ratio, 10), "NO": round(1.0 - yes_ratio, 10)},
        })
    return records


def gold_from_dataset_hard_2_2(
        data: list[dict], ids: list[str], annotator_threshold: int = Config.HARD_THRESHOLD_2_2
) -> list[dict]:
    """
    Build hard gold for subtask 2.2.

    Maps "-" to NO, skips UNKNOWN. Selects the class with > threshold annotators.
    """
    id_to_item = {str(item.get("id_EXIST", "")): item for item in data}
    records = []
    for mid in ids:
        item = id_to_item.get(str(mid))
        if item is None:
            continue
        raw_labels = item.get("labels_task2_2", [])
        counts = Counter()
        for l in raw_labels:
            lu = l.upper().strip()
            if lu in ("UNKNOWN", ""):
                continue
            if lu == "-":
                lu = "NO"
            counts[lu] += 1

        if not counts:
            continue

        # Find the class exceeding the threshold
        gold_label = None
        for label, count in counts.most_common():
            if count > annotator_threshold and label in Task22.LABEL_TO_INDEX:
                gold_label = label
                break

        if gold_label is None:
            continue

        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": gold_label})
    return records


def gold_from_dataset_soft_2_2(data: list[dict], ids: list[str]) -> list[dict]:
    """Build soft gold for subtask 2.2."""
    id_to_item = {str(item.get("id_EXIST", "")): item for item in data}
    records = []
    for mid in ids:
        item = id_to_item.get(str(mid))
        if item is None:
            continue
        raw_labels = item.get("labels_task2_2", [])
        counts = Counter()
        total = 0
        for l in raw_labels:
            lu = l.upper().strip()
            if lu in ("UNKNOWN", ""):
                continue
            if lu == "-":
                lu = "NO"
            if lu in Task22.LABEL_TO_INDEX:
                counts[lu] += 1
                total += 1

        if total == 0:
            continue

        value = {label: round(counts.get(label, 0) / total, 10) for label in Task22.LABELS}
        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": value})
    return records


def gold_from_dataset_hard_2_3(
        data: list[dict], ids: list[str], annotator_threshold: int = Config.HARD_THRESHOLD_2_3
) -> list[dict]:
    """
    Build hard gold for subtask 2.3 (multi-label).

    Each annotator provides a list of categories. A category is selected if > threshold annotators chose it.
    """
    id_to_item = {str(item.get("id_EXIST", "")): item for item in data}
    records = []
    for mid in ids:
        item = id_to_item.get(str(mid))
        if item is None:
            continue
        labels_list = item.get("labels_task2_3", [])
        counts = Counter()
        n_valid = 0

        for annotator_labels in labels_list:
            if isinstance(annotator_labels, str):
                annotator_labels = [annotator_labels]
            valid_for_ann = []
            for l in annotator_labels:
                lu = l.upper().strip()
                if lu in ("UNKNOWN", ""):
                    continue
                if lu == "-":
                    lu = "NO"
                if lu in Task23.LABEL_TO_INDEX:
                    valid_for_ann.append(lu)
            if valid_for_ann:
                n_valid += 1
                for vl in valid_for_ann:
                    counts[vl] += 1

        if n_valid == 0:
            continue

        selected = [label for label, cnt in counts.items() if cnt > annotator_threshold]
        if not selected:
            continue

        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": selected})
    return records


def gold_from_dataset_soft_2_3(data: list[dict], ids: list[str]) -> list[dict]:
    """Build soft gold for subtask 2.3 (multi-label proportions)."""
    id_to_item = {str(item.get("id_EXIST", "")): item for item in data}
    records = []
    for mid in ids:
        item = id_to_item.get(str(mid))
        if item is None:
            continue
        labels_list = item.get("labels_task2_3", [])
        counts = Counter()
        n_valid = 0

        for annotator_labels in labels_list:
            if isinstance(annotator_labels, str):
                annotator_labels = [annotator_labels]
            valid_for_ann = []
            for l in annotator_labels:
                lu = l.upper().strip()
                if lu in ("UNKNOWN", ""):
                    continue
                if lu == "-":
                    lu = "NO"
                if lu in Task23.LABEL_TO_INDEX:
                    valid_for_ann.append(lu)
            if valid_for_ann:
                n_valid += 1
                for vl in valid_for_ann:
                    counts[vl] += 1

        if n_valid == 0:
            continue

        value = {label: round(counts.get(label, 0) / n_valid, 10) for label in Task23.LABELS}
        records.append({"test_case": Config.TEST_CASE, "id": str(mid), "value": value})
    return records


def _run_pyevall(
        pred_records: list[dict],
        gold_records: list[dict],
        metrics: list[str],
        hierarchy: dict | None = None,
) -> dict[str, float]:
    """Generic PyEvALL evaluation runner."""
    if not pred_records or not gold_records:
        return {m: 0.0 for m in metrics}

    gold_ids = {r["id"] for r in gold_records}
    pred_filtered = [r for r in pred_records if r["id"] in gold_ids]
    if not pred_filtered:
        return {m: 0.0 for m in metrics}

    pred_path = _write_tmp_json(pred_filtered)
    gold_path = _write_tmp_json(gold_records)
    try:
        evaluator = PyEvALLEvaluation()
        params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}
        if hierarchy is not None:
            params[PyEvALLUtils.PARAM_HIERARCHY] = hierarchy
        report = evaluator.evaluate(pred_path, gold_path, metrics, **params)
        return _extract_metrics(report.report)
    except Exception as e:
        print(f"PyEvALL evaluation failed: {e}")
        return {m: 0.0 for m in metrics}
    finally:
        Path(pred_path).unlink(missing_ok=True)
        Path(gold_path).unlink(missing_ok=True)


def evaluate_hard_2_1(pred_records, gold_records) -> dict[str, float]:
    return _run_pyevall(pred_records, gold_records, ["ICM", "ICMNorm", "FMeasure"])


def evaluate_soft_2_1(pred_records, gold_records) -> dict[str, float]:
    return _run_pyevall(pred_records, gold_records, ["ICMSoft", "ICMSoftNorm", "CrossEntropy"])


def evaluate_hard_2_2(pred_records, gold_records) -> dict[str, float]:
    icm = _run_pyevall(
        pred_records, gold_records,
        ["ICM", "ICMNorm"],
        hierarchy=Task22.HIERARCHY,
    )
    f1 = _run_pyevall(pred_records, gold_records, ["FMeasure"])
    return {**icm, **f1}


def evaluate_soft_2_2(pred_records, gold_records) -> dict[str, float]:
    icm = _run_pyevall(
        pred_records, gold_records,
        ["ICMSoft", "ICMSoftNorm"],
        hierarchy=Task22.HIERARCHY,
    )
    ce = _run_pyevall(pred_records, gold_records, ["CrossEntropy"])
    return {**icm, **ce}


def evaluate_hard_2_3(pred_records, gold_records) -> dict[str, float]:
    icm = _run_pyevall(
        pred_records, gold_records,
        ["ICM", "ICMNorm"],
        hierarchy=Task23.HIERARCHY,
    )
    f1 = _run_pyevall(pred_records, gold_records, ["FMeasure"])
    return {**icm, **f1}


def evaluate_soft_2_3(pred_records, gold_records) -> dict[str, float]:
    return _run_pyevall(
        pred_records, gold_records,
        ["ICMSoft", "ICMSoftNorm"],
        hierarchy=Task23.HIERARCHY,
    )


def evaluate_epoch_2_1(
        ids: list[str], probs_yes: list[float], dataset_data: list[dict], threshold: float = 0.5
) -> dict[str, float]:
    """Run both hard and soft evaluation for subtask 2.1."""
    pred_hard = probs_to_pyevall_hard_2_1(ids, probs_yes, threshold)
    pred_soft = probs_to_pyevall_soft_2_1(ids, probs_yes)
    gold_hard = gold_from_dataset_hard_2_1(dataset_data, ids)
    gold_soft = gold_from_dataset_soft_2_1(dataset_data, ids)

    combined = {}
    for k, v in evaluate_hard_2_1(pred_hard, gold_hard).items():
        combined[f"2.1/hard/{k}"] = v
    for k, v in evaluate_soft_2_1(pred_soft, gold_soft).items():
        combined[f"2.1/soft/{k}"] = v
    return combined


def evaluate_epoch_2_2(
        ids: list[str], probs: list[list[float]], dataset_data: list[dict]
) -> dict[str, float]:
    """Run both hard and soft evaluation for subtask 2.2."""
    pred_hard = probs_to_pyevall_hard_2_2(ids, probs)
    pred_soft = probs_to_pyevall_soft_2_2(ids, probs)
    gold_hard = gold_from_dataset_hard_2_2(dataset_data, ids)
    gold_soft = gold_from_dataset_soft_2_2(dataset_data, ids)

    combined = {}
    for k, v in evaluate_hard_2_2(pred_hard, gold_hard).items():
        combined[f"2.2/hard/{k}"] = v
    for k, v in evaluate_soft_2_2(pred_soft, gold_soft).items():
        combined[f"2.2/soft/{k}"] = v
    return combined


def evaluate_epoch_2_3(
        ids: list[str], probs: list[list[float]], dataset_data: list[dict], threshold: float = 0.5
) -> dict[str, float]:
    """Run both hard and soft evaluation for subtask 2.3."""
    pred_hard = probs_to_pyevall_hard_2_3(ids, probs, threshold)
    pred_soft = probs_to_pyevall_soft_2_3(ids, probs)
    gold_hard = gold_from_dataset_hard_2_3(dataset_data, ids)
    gold_soft = gold_from_dataset_soft_2_3(dataset_data, ids)

    combined = {}
    for k, v in evaluate_hard_2_3(pred_hard, gold_hard).items():
        combined[f"2.3/hard/{k}"] = v
    for k, v in evaluate_soft_2_3(pred_soft, gold_soft).items():
        combined[f"2.3/soft/{k}"] = v
    return combined


def evaluate_epoch_all(
        ids: list[str],
        probs_2_1: list[float] | None = None,
        probs_2_2: list[list[float]] | None = None,
        probs_2_3: list[list[float]] | None = None,
        dataset_data: list[dict] = None,
        threshold_2_1: float = 0.5,
        threshold_2_3: float = 0.5,
) -> dict[str, float]:
    """Run evaluation for all active subtasks and merge results."""
    combined = {}
    if probs_2_1 is not None:
        combined.update(evaluate_epoch_2_1(ids, probs_2_1, dataset_data, threshold_2_1))
    if probs_2_2 is not None:
        combined.update(evaluate_epoch_2_2(ids, probs_2_2, dataset_data))
    if probs_2_3 is not None:
        combined.update(evaluate_epoch_2_3(ids, probs_2_3, dataset_data, threshold_2_3))
    return combined


def save_submission_2_1(
        ids: list[str], probs_yes: list[float], save_dir: str | Path,
        team_name: str, run_id: int = 1, threshold: float = 0.5,
) -> tuple[Path, Path]:
    """Generate official submission files for subtask 2.1."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    hard_records = probs_to_pyevall_hard_2_1(ids, probs_yes, threshold)
    soft_records = probs_to_pyevall_soft_2_1(ids, probs_yes)

    hard_path = save_dir / f"task2_1_hard_{team_name}_{run_id}.json"
    soft_path = save_dir / f"task2_1_soft_{team_name}_{run_id}.json"
    with open(hard_path, "w") as f:
        json.dump(hard_records, f, indent=2)
    with open(soft_path, "w") as f:
        json.dump(soft_records, f, indent=2)
    print(f"Saved 2.1 hard -> {hard_path}")
    print(f"Saved 2.1 soft -> {soft_path}")
    return hard_path, soft_path


def save_submission_2_2(
        ids: list[str], probs: list[list[float]], save_dir: str | Path,
        team_name: str, run_id: int = 1,
) -> tuple[Path, Path]:
    """Generate official submission files for subtask 2.2."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    hard_records = probs_to_pyevall_hard_2_2(ids, probs)
    soft_records = probs_to_pyevall_soft_2_2(ids, probs)

    hard_path = save_dir / f"task2_2_hard_{team_name}_{run_id}.json"
    soft_path = save_dir / f"task2_2_soft_{team_name}_{run_id}.json"
    with open(hard_path, "w") as f:
        json.dump(hard_records, f, indent=2)
    with open(soft_path, "w") as f:
        json.dump(soft_records, f, indent=2)
    print(f"Saved 2.2 hard -> {hard_path}")
    print(f"Saved 2.2 soft -> {soft_path}")
    return hard_path, soft_path


def save_submission_2_3(
        ids: list[str], probs: list[list[float]], save_dir: str | Path,
        team_name: str, run_id: int = 1, threshold: float = 0.5,
) -> tuple[Path, Path]:
    """Generate official submission files for subtask 2.3."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    hard_records = probs_to_pyevall_hard_2_3(ids, probs, threshold)
    soft_records = probs_to_pyevall_soft_2_3(ids, probs)

    hard_path = save_dir / f"task2_3_hard_{team_name}_{run_id}.json"
    soft_path = save_dir / f"task2_3_soft_{team_name}_{run_id}.json"
    with open(hard_path, "w") as f:
        json.dump(hard_records, f, indent=2)
    with open(soft_path, "w") as f:
        json.dump(soft_records, f, indent=2)
    print(f"Saved 2.3 hard -> {hard_path}")
    print(f"Saved 2.3 soft -> {soft_path}")
    return hard_path, soft_path
