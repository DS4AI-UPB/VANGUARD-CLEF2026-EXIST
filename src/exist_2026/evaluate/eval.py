import json
from pathlib import Path

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

from exist_2026.evaluate.eval_utils import _write_tmp_json, _extract_metrics

TEST_CASE = "EXIST2025"


def probs_to_pyevall_hard(ids: list[str], probs_yes: list[float], threshold: float = 0.5) -> list[dict]:
    """Convert predicted P(YES) values to hard-label PyEvALL format."""
    records = []
    for meme_id, p in zip(ids, probs_yes):
        label = "YES" if p >= threshold else "NO"
        records.append({"test_case": TEST_CASE, "id": str(meme_id), "value": label})
    return records


def probs_to_pyevall_soft(ids: list[str], probs_yes: list[float]) -> list[dict]:
    """Convert predicted P(YES) values to soft-label PyEvALL format."""
    records = []
    for meme_id, p in zip(ids, probs_yes):
        records.append({
            "test_case": TEST_CASE,
            "id": str(meme_id),
            "value": {"YES": round(float(p), 10), "NO": round(1.0 - float(p), 10)},
        })
    return records


def gold_from_dataset_hard(
        data: list[dict], ids: list[str], annotator_threshold: int = 3
) -> list[dict]:
    """
    Build a hard gold standard from annotator labels for subtask 2.1.

    Per the guidelines, the class annotated by **more than** `annotator_threshold`
    annotators is selected. Items with no majority are excluded.
    """
    id_to_item = {item.get("id_EXIST", ""): item for item in data}
    records = []
    for meme_id in ids:
        item = id_to_item.get(str(meme_id))
        if item is None:
            continue
        labels = [l.upper() for l in item.get("labels_task2_1", []) if l.upper() != "UNKNOWN"]
        if not labels:
            continue
        yes_count = sum(1 for l in labels if l == "YES")
        no_count = sum(1 for l in labels if l == "NO")
        if yes_count > annotator_threshold:
            gold_label = "YES"
        elif no_count > annotator_threshold:
            gold_label = "NO"
        else:
            # no majority then is excluded from hard evaluation
            continue
        records.append({"test_case": TEST_CASE, "id": str(meme_id), "value": gold_label})
    return records


def gold_from_dataset_soft(data: list[dict], ids: list[str]) -> list[dict]:
    """Build a soft gold standard from annotator label distributions for subtask 2.1."""
    id_to_item = {item.get("id_EXIST", ""): item for item in data}
    records = []
    for meme_id in ids:
        item = id_to_item.get(str(meme_id))
        if item is None:
            continue
        labels = [l.upper() for l in item.get("labels_task2_1", []) if l.upper() != "UNKNOWN"]
        if not labels:
            continue
        yes_ratio = sum(1 for l in labels if l == "YES") / len(labels)
        records.append({
            "test_case": TEST_CASE,
            "id": str(meme_id),
            "value": {"YES": round(yes_ratio, 10), "NO": round(1.0 - yes_ratio, 10)},
        })
    return records


def evaluate_hard(
        pred_records: list[dict],
        gold_records: list[dict],
) -> dict[str, float]:
    """Run hard-hard evaluation: ICM, ICMNorm, FMeasure."""
    if not pred_records or not gold_records:
        print("Empty predictions or gold for hard evaluation, returning zeros.")
        return {"ICM": 0.0, "ICMNorm": 0.0, "FMeasure": 0.0}

    # Align: only evaluate IDs present in both
    gold_ids = {r["id"] for r in gold_records}
    pred_filtered = [r for r in pred_records if r["id"] in gold_ids]

    if not pred_filtered:
        print("No overlapping IDs between predictions and gold.")
        return {"ICM": 0.0, "ICMNorm": 0.0, "FMeasure": 0.0}

    pred_path = _write_tmp_json(pred_filtered)
    gold_path = _write_tmp_json(gold_records)
    try:
        evaluator = PyEvALLEvaluation()
        params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}
        metrics = ["ICM", "ICMNorm", "FMeasure"]
        report = evaluator.evaluate(pred_path, gold_path, metrics, **params)
        return _extract_metrics(report.report)
    except Exception as e:
        print(f"PyEvALL hard evaluation failed: {e}")
        return {"ICM": 0.0, "ICMNorm": 0.0, "FMeasure": 0.0}
    finally:
        Path(pred_path).unlink(missing_ok=True)
        Path(gold_path).unlink(missing_ok=True)


def evaluate_soft(
        pred_records: list[dict],
        gold_records: list[dict],
) -> dict[str, float]:
    """Run soft-soft evaluation: ICMSoft, ICMSoftNorm, CrossEntropy."""
    if not pred_records or not gold_records:
        print("Empty predictions or gold for soft evaluation, returning zeros.")
        return {"ICMSoft": 0.0, "ICMSoftNorm": 0.0, "CrossEntropy": 0.0}

    gold_ids = {r["id"] for r in gold_records}
    pred_filtered = [r for r in pred_records if r["id"] in gold_ids]

    if not pred_filtered:
        print("No overlapping IDs between predictions and gold (soft).")
        return {"ICMSoft": 0.0, "ICMSoftNorm": 0.0, "CrossEntropy": 0.0}

    pred_path = _write_tmp_json(pred_filtered)
    gold_path = _write_tmp_json(gold_records)
    try:
        evaluator = PyEvALLEvaluation()
        params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}
        metrics = ["ICMSoft", "ICMSoftNorm", "CrossEntropy"]
        report = evaluator.evaluate(pred_path, gold_path, metrics, **params)
        return _extract_metrics(report.report)
    except Exception as e:
        print(f"PyEvALL soft evaluation failed: {e}")
        return {"ICMSoft": 0.0, "ICMSoftNorm": 0.0, "CrossEntropy": 0.0}
    finally:
        Path(pred_path).unlink(missing_ok=True)
        Path(gold_path).unlink(missing_ok=True)


def evaluate_epoch(
        ids: list[str],
        probs_yes: list[float],
        dataset_data: list[dict],
        threshold: float = 0.5,
) -> dict[str, float]:
    pred_hard = probs_to_pyevall_hard(ids, probs_yes, threshold=threshold)
    pred_soft = probs_to_pyevall_soft(ids, probs_yes)

    gold_hard = gold_from_dataset_hard(dataset_data, ids)
    gold_soft = gold_from_dataset_soft(dataset_data, ids)

    hard_metrics = evaluate_hard(pred_hard, gold_hard)
    soft_metrics = evaluate_soft(pred_soft, gold_soft)

    combined = {}
    for k, v in hard_metrics.items():
        combined[f"hard/{k}"] = v
    for k, v in soft_metrics.items():
        combined[f"soft/{k}"] = v

    return combined


def save_submission(
        ids: list[str],
        probs_yes: list[float],
        save_dir: str | Path,
        team_name: str,
        run_id: int = 1,
        threshold: float = 0.5,
) -> tuple[Path, Path]:
    """Generate official submission files for subtask 2.1 (hard + soft)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    hard_records = probs_to_pyevall_hard(ids, probs_yes, threshold=threshold)
    soft_records = probs_to_pyevall_soft(ids, probs_yes)

    hard_path = save_dir / f"task2_1_hard_{team_name}_{run_id}.json"
    soft_path = save_dir / f"task2_1_soft_{team_name}_{run_id}.json"

    with open(hard_path, "w") as f:
        json.dump(hard_records, f, indent=2)
    with open(soft_path, "w") as f:
        json.dump(soft_records, f, indent=2)

    print(f"Saved hard submission → {hard_path}")
    print(f"Saved soft submission → {soft_path}")
    return hard_path, soft_path
