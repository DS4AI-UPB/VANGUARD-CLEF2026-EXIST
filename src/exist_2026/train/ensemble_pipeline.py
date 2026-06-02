import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exist_2026.consts.task_config import Config, Task22, Task23


def extract_classical_features(item: dict) -> list[float]:
    """
    Extract handcrafted features from a single meme item.

    Returns a fixed-size vector covering:
        - Stylometry (4): text length, word count, caps ratio, punctuation ratio
        - Demographics (2): female annotator ratio, older annotator ratio
        - Eye-tracking (4): log reaction time, mean pupil diameter, fixation count, saccade count
        - Heart rate (2): mean HR, HR std
        - EEG (5): mean bandpower for delta, theta, alpha, beta, gamma
    """
    features = []

    # Stylometry
    text = item.get("clean_text", "")
    features.append(len(text))
    features.append(len(text.split()))
    features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))
    features.append(len(re.findall(r"[!?¡¿]", text)) / max(len(text), 1))

    # Demographics
    genders = item.get("gender_annotators", [])
    ages = item.get("age_annotators", [])
    features.append(sum(1 for g in genders if g == "F") / max(len(genders), 1))
    features.append(sum(1 for a in ages if a == "46+") / max(len(ages), 1))

    # Eye-tracking
    sensorial = item.get("sensorial", {})
    modalities = sensorial.get("modalities", {})
    et_data = modalities.get("ET", {}).get("by_user", {})

    rt_vals = [float(v["reaction_time"]) for v in et_data.values()
               if v.get("reaction_time") is not None]
    features.append(np.log1p(np.mean(rt_vals)) if rt_vals else np.log1p(12000.0))

    pupil_left = [float(v["3d_eye_states_pupil diameter left [mm]_mean"]) for v in et_data.values()
                  if v.get("3d_eye_states_pupil diameter left [mm]_mean") is not None]
    features.append(np.mean(pupil_left) if pupil_left else 3.0)

    fix_counts = [float(v["fixations_count"]) for v in et_data.values()
                  if v.get("fixations_count") is not None]
    features.append(np.mean(fix_counts) if fix_counts else 34.0)

    sac_counts = [float(v["saccades_count"]) for v in et_data.values()
                  if v.get("saccades_count") is not None]
    features.append(np.mean(sac_counts) if sac_counts else 33.0)

    # Heart rate
    hr_data = modalities.get("HR", {}).get("by_user", {})
    hr_means = [float(v["garmin_hr_mean"]) for v in hr_data.values()
                if v.get("garmin_hr_mean") is not None]
    features.append(np.mean(hr_means) if hr_means else 75.0)

    hr_stds = [float(v["garmin_hr_std"]) for v in hr_data.values()
               if v.get("garmin_hr_std") is not None]
    features.append(np.mean(hr_stds) if hr_stds else 3.0)

    # EEG (mean bandpower across all channels per band)
    eeg_data = modalities.get("EEG", {}).get("by_user", {})
    for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
        vals = [
            float(user_eeg[f"EXG_Channel_{ch}_{band}_power"])
            for user_eeg in eeg_data.values()
            for ch in range(16)
            if user_eeg.get(f"EXG_Channel_{ch}_{band}_power") is not None
        ]
        features.append(np.mean(vals) if vals else 0.0)

    return features


def _get_hard_label_2_1(item: dict) -> int | None:
    """Majority-vote binary label. Returns 1 (YES) or 0 (NO), or None if ambiguous."""
    labels = [l.upper() for l in item.get("labels_task2_1", [])
              if l.upper() not in Config.SKIP_LABELS]
    if not labels:
        return None
    yes = sum(1 for l in labels if l == "YES")
    return 1 if yes / len(labels) >= 0.5 else 0


def _get_hard_label_2_2(item: dict) -> int | None:
    """Majority-vote 3-class label index (NO=0, DIRECT=1, JUDGEMENTAL=2)."""
    raw = item.get("labels_task2_2", [])
    counts = Counter()
    for l in raw:
        lu = l.upper().strip()
        if lu in ("UNKNOWN", ""):
            continue
        if lu == "-":
            lu = "NO"
        if lu in Task22.LABEL_TO_INDEX:
            counts[lu] += 1
    if not counts:
        return None
    majority = counts.most_common(1)[0][0]
    return Task22.LABEL_TO_INDEX[majority]


def _get_hard_label_2_3(item: dict) -> list[int] | None:
    """Multi-label binary vector of length 6. Returns None if no valid labels."""
    labels_list = item.get("labels_task2_3", [])
    counts = Counter()
    n_valid = 0
    for ann_labels in labels_list:
        if isinstance(ann_labels, str):
            ann_labels = [ann_labels]
        valid = []
        for l in ann_labels:
            lu = l.upper().strip()
            if lu in ("UNKNOWN", ""):
                continue
            if lu == "-":
                lu = "NO"
            if lu in Task23.LABEL_TO_INDEX:
                valid.append(lu)
        if valid:
            n_valid += 1
            for v in valid:
                counts[v] += 1
    if n_valid == 0:
        return None
    vec = [0] * Task23.NUM_CLASSES
    for label, cnt in counts.items():
        if cnt > 0:
            vec[Task23.LABEL_TO_INDEX[label]] = 1
    return vec


LABEL_EXTRACTORS = {
    "2.1": _get_hard_label_2_1,
    "2.2": _get_hard_label_2_2,
    "2.3": _get_hard_label_2_3,
}


def build_feature_matrix(data: list[dict], task: str):
    """
    Build X, y matrices for a given task.

    :returns:
        - x: (n_samples, n_features)
        - y: for 2.1/2.2 -> (n_samples,) int array
           for 2.3 -> (n_samples, 6) binary array
        - valid_indices: which items from `data` had valid labels
    """
    extractor = LABEL_EXTRACTORS[task]
    X, y, indices = [], [], []
    for i, item in enumerate(data):
        label = extractor(item)
        if label is None:
            continue
        X.append(extract_classical_features(item))
        y.append(label)
        indices.append(i)
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    return X, y, indices


def train_svm(X_train: np.ndarray, y_train: np.ndarray, task: str):
    """Train an SVM appropriate for the task type."""
    if task == "2.3":
        # Multi-label: one SVM per class via OVR
        base = make_pipeline(
            StandardScaler(),
            SVC(probability=True, kernel="rbf", C=1.0, class_weight="balanced"),
        )
        model = OneVsRestClassifier(base)
    else:
        # Binary (2.1) or multiclass (2.2)
        model = make_pipeline(
            StandardScaler(),
            SVC(probability=True, kernel="rbf", C=1.0, class_weight="balanced"),
        )
    model.fit(X_train, y_train)
    return model


def get_svm_probs(model, X: np.ndarray, task: str):
    """
    Get probability predictions from the SVM in the same format as DL outputs.

    :return:
        - 2.1: list[float] -> P(YES) per sample
        - 2.2: list[list[float]] -> [P(NO), P(DIRECT), P(JUDGEMENTAL)] per sample
        - 2.3: list[list[float]] -> [P(c) for c in 6 classes] per sample
    """
    proba = model.predict_proba(X)
    if task == "2.1":
        # SVM classes are [0, 1]; we want P(YES) = P(class=1)
        if model.classes_[-1] == 1 if hasattr(model, "classes_") else True:
            return proba[:, 1].tolist()
        clf = model[-1] if hasattr(model, "__getitem__") else model
        if hasattr(clf, "classes_") and list(clf.classes_) == [0, 1]:
            return proba[:, 1].tolist()
        return proba[:, 1].tolist()
    elif task == "2.2":
        # Ensure ordering matches [NO=0, DIRECT=1, JUDGEMENTAL=2]
        return proba.tolist()
    elif task == "2.3":
        return proba.tolist()


def blend_probs_2_1(
        dl_probs: list[float],
        svm_probs: list[float],
        dl_weight: float,
) -> list[float]:
    """Weighted average of P(YES) from DL and SVM."""
    svm_w = 1.0 - dl_weight
    return [dl_weight * d + svm_w * s for d, s in zip(dl_probs, svm_probs)]


def blend_probs_multiclass(
        dl_probs: list[list[float]],
        svm_probs: list[list[float]],
        dl_weight: float,
) -> list[list[float]]:
    """Weighted average of class probability vectors."""
    svm_w = 1.0 - dl_weight
    blended = []
    for d, s in zip(dl_probs, svm_probs):
        blended.append([dl_weight * di + svm_w * si for di, si in zip(d, s)])
    return blended


def optimize_blend_2_1(
        dl_probs: list[float],
        svm_probs: list[float],
        y_true: np.ndarray,
        dl_range=(0.3, 1.01, 0.05),
        thresh_range=(0.25, 0.75, 0.02),
) -> tuple[float, float, float]:
    """Grid search for best DL weight and threshold for task 2.1."""
    best_f1, best_w, best_t = 0.0, 0.7, 0.5
    for w in np.arange(*dl_range):
        blended = blend_probs_2_1(dl_probs, svm_probs, w)
        for t in np.arange(*thresh_range):
            preds = [1 if p >= t else 0 for p in blended]
            f1 = f1_score(y_true, preds, average="macro")
            if f1 > best_f1:
                best_f1, best_w, best_t = f1, w, t
    return best_w, best_t, best_f1


def optimize_blend_2_2(
        dl_probs: list[list[float]],
        svm_probs: list[list[float]],
        y_true: np.ndarray,
        dl_range=(0.3, 1.01, 0.05),
) -> tuple[float, float]:
    """Grid search for best DL weight for task 2.2 (argmax for prediction)."""
    best_f1, best_w = 0.0, 0.7
    for w in np.arange(*dl_range):
        blended = blend_probs_multiclass(dl_probs, svm_probs, w)
        preds = [np.argmax(p) for p in blended]
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


def optimize_blend_2_3(
        dl_probs: list[list[float]],
        svm_probs: list[list[float]],
        y_true: np.ndarray,
        dl_range=(0.3, 1.01, 0.05),
        thresh_range=(0.25, 0.75, 0.05),
) -> tuple[float, float, float]:
    """Grid search for best DL weight and threshold for task 2.3 (multi-label)."""
    best_f1, best_w, best_t = 0.0, 0.7, 0.5
    for w in np.arange(*dl_range):
        blended = blend_probs_multiclass(dl_probs, svm_probs, w)
        for t in np.arange(*thresh_range):
            preds = []
            for p in blended:
                row = [0] * len(p)
                for i, val in enumerate(p):
                    if Task23.LABELS[i] == "NO":
                        continue
                    if val >= t:
                        row[i] = 1
                if sum(row) == 0:
                    row[0] = 1  # default to NO
                preds.append(row)
            f1 = f1_score(y_true, np.array(preds), average="macro")
            if f1 > best_f1:
                best_f1, best_w, best_t = f1, w, t
    return best_w, best_t, best_f1


class EnsembleResult:
    def __init__(self, task: str):
        self.task = task
        self.svm_model = None
        self.dl_weight: float = 0.7
        self.threshold: float = 0.5
        self.best_f1: float = 0.0

    def save(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.svm_model, save_dir / f"svm_{self.task.replace('.', '_')}.joblib")
        meta = {
            "task": self.task,
            "dl_weight": self.dl_weight,
            "threshold": self.threshold,
            "best_f1": self.best_f1,
        }
        with open(save_dir / f"ensemble_meta_{self.task.replace('.', '_')}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved ensemble artifacts for task {self.task} → {save_dir}")


def run_ensemble_for_task(
        task: str,
        train_data: list[dict],
        val_data: list[dict],
        val_ids: list[str],
        dl_val_probs,
        save_dir: Path,
) -> EnsembleResult:
    """
    Train SVM, optimize blend, return EnsembleResult.
 
    :param task: "2.1", "2.2", or "2.3"
    :param train_data: list of raw data dicts for training split
    :param val_data: list of raw data dicts for validation split
    :param val_ids: list of meme IDs in validation set (same order as dl_val_probs)
    :param dl_val_probs: DL predictions on val set:
                                - 2.1: list[float] (P(YES))
                                - 2.2: list[list[float]] (3-class)
                                - 2.3: list[list[float]] (6-class)
    :param save_dir: where to save SVM model and blend metadata
    """
    result = EnsembleResult(task)

    print(f"\n\tEnsemble for task {task}")

    X_train, y_train, _ = build_feature_matrix(train_data, task)
    print(f"  SVM train samples: {len(X_train)}")

    if len(X_train) == 0:
        print(f"\tWARNING: no valid training samples for task {task}, skipping SVM")
        result.dl_weight = 1.0
        result.save(save_dir)
        return result

    svm = train_svm(X_train, y_train, task)
    result.svm_model = svm

    id_to_item = {str(d.get("id_EXIST", "")): d for d in val_data}
    X_val_list, y_val_list, valid_mask = [], [], []
    extractor = LABEL_EXTRACTORS[task]
    for mid in val_ids:
        item = id_to_item.get(str(mid))
        if item is None:
            valid_mask.append(False)
            continue
        label = extractor(item)
        if label is None:
            valid_mask.append(False)
            continue
        X_val_list.append(extract_classical_features(item))
        y_val_list.append(label)
        valid_mask.append(True)

    if not X_val_list:
        print(f"\tWARNING: no valid val samples for task {task}, using DL only")
        result.dl_weight = 1.0
        result.save(save_dir)
        return result

    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)
    svm_val_probs = get_svm_probs(svm, X_val, task)

    dl_val_filtered = [p for p, m in zip(dl_val_probs, valid_mask) if m]

    print(f"\tSVM val samples: {len(X_val)}")

    if task == "2.1":
        w, t, f1 = optimize_blend_2_1(dl_val_filtered, svm_val_probs, y_val)
        result.dl_weight, result.threshold, result.best_f1 = w, t, f1
        print(f"\tBlend: {w * 100:.0f}% DL + {(1 - w) * 100:.0f}% SVM | thresh={t:.2f} | F1={f1:.4f}")

    elif task == "2.2":
        w, f1 = optimize_blend_2_2(dl_val_filtered, svm_val_probs, y_val)
        result.dl_weight, result.best_f1 = w, f1
        print(f"\tBlend: {w * 100:.0f}% DL + {(1 - w) * 100:.0f}% SVM | F1={f1:.4f}")

    elif task == "2.3":
        w, t, f1 = optimize_blend_2_3(dl_val_filtered, svm_val_probs, y_val)
        result.dl_weight, result.threshold, result.best_f1 = w, t, f1
        print(f"\tBlend: {w * 100:.0f}% DL + {(1 - w) * 100:.0f}% SVM | thresh={t:.2f} | F1={f1:.4f}")

    result.save(save_dir)
    return result


def apply_ensemble_to_test(
        task: str,
        ensemble: EnsembleResult,
        test_data: list[dict],
        test_ids: list[str],
        dl_test_probs,
) -> list:
    """
    Apply trained ensemble to test predictions.

    Returns blended probabilities in the same format as DL predictions.
    """
    if ensemble.svm_model is None or ensemble.dl_weight >= 1.0:
        return dl_test_probs

    id_to_item = {str(d.get("id_EXIST", "")): d for d in test_data}
    X_test = []
    for mid in test_ids:
        item = id_to_item.get(str(mid))
        if item is not None:
            X_test.append(extract_classical_features(item))
        else:
            X_test.append([0.0] * 17)
    X_test = np.array(X_test)

    svm_test_probs = get_svm_probs(ensemble.svm_model, X_test, task)

    if task == "2.1":
        return blend_probs_2_1(dl_test_probs, svm_test_probs, ensemble.dl_weight)
    else:
        return blend_probs_multiclass(dl_test_probs, svm_test_probs, ensemble.dl_weight)
