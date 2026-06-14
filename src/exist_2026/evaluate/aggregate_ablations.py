import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

from exist_2026.path_manager import PathManager


def load_all_seeds(ablation_root: Path) -> dict[str, list[dict]]:
    results = {}
    for sub in sorted(ablation_root.iterdir()):
        if not sub.is_dir():
            continue
        seed_dicts = []
        for seed_dir in sorted(sub.iterdir()):
            if not (seed_dir.is_dir() and seed_dir.name.startswith("seed_")):
                continue
            path = seed_dir / "val_metrics_full.json"
            if path.exists():
                with open(path) as f:
                    seed_dicts.append(json.load(f))
        if seed_dicts:
            results[sub.name] = seed_dicts
            print(f"[load] {sub.name}: {len(seed_dicts)} seed(s)")
    return results


def get_values(seed_dicts: list[dict], task: str, kind: str, key: str) -> list[float]:
    values = []
    for d in seed_dicts:
        if "All" not in d:
            continue
        v = d["All"].get(f"{task}/{kind}/{key}")
        if v is not None:
            values.append(v)
    return values


# (task, kind, key, label). Higher-is-better for ICM/F1; CE is lower-is-better
# but the test is symmetric, so direction is only used for the reported sign.
METRICS = [
    ("2.1", "hard", "ICMNorm", "2.1 ICM-N"),
    ("2.1", "hard", "FMeasure", "2.1 F1"),
    ("2.1", "soft", "ICMSoftNorm", "2.1 ICM-SN"),
    ("2.2", "hard", "ICMNorm", "2.2 ICM-N"),
    ("2.2", "hard", "FMeasure", "2.2 F1"),
    ("2.2", "soft", "ICMSoftNorm", "2.2 ICM-SN"),
    ("2.3", "hard", "ICMNorm", "2.3 ICM-N"),
    ("2.3", "hard", "FMeasure", "2.3 F1"),
    ("2.3", "soft", "ICMSoftNorm", "2.3 ICM-SN"),
]

ROW_ORDER = [
    "no_film", "no_description", "no_image", "text_only",
    "no_contrastive", "no_aux_head", "only_2_1", "only_2_2", "only_2_3",
]


def _holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Return a boolean "reject null" list, Holm-Bonferroni step-down."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def print_significance_summary(results: dict[str, list[dict]], alpha: float = 0.05) -> None:
    if "baseline" not in results:
        print("No baseline config found... Cannot run significance tests.")
        return
    base = results["baseline"]

    rows = []  # (config, metric_label, delta, welch_p, mwu_p)
    for name in ROW_ORDER:
        if name not in results:
            continue
        abl = results[name]
        for task, kind, key, label in METRICS:
            b = get_values(base, task, kind, key)
            a = get_values(abl, task, kind, key)
            if len(b) < 2 or len(a) < 2:
                continue
            if np.std(b) == 0 and np.std(a) == 0 and np.mean(a) == np.mean(b):
                continue
            delta = float(np.mean(a) - np.mean(b))
            try:
                welch_p = float(ttest_ind(a, b, equal_var=False).pvalue)
            except Exception:
                welch_p = float("nan")
            try:
                mwu_p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                mwu_p = float("nan")
            rows.append((name, label, delta, welch_p, mwu_p))

    welch_ps = [r[3] for r in rows]
    reject = _holm_bonferroni(welch_ps, alpha=alpha)

    print("\n" + "=" * 78)
    print("Significance vs baseline (Welch t-test + Mann-Whitney U, n seeds)")
    print(f"Holm-Bonferroni family size = {len(rows)}, alpha = {alpha}")
    print("=" * 78)
    print(f"{'config':<16}{'metric':<12}{'delta':>9}{'welch_p':>9}{'mwu_p':>8}  holm")
    print("-" * 78)
    for (name, label, delta, wp, mp), rej in zip(rows, reject):
        mark = "REJECT*" if rej else ""
        print(f"{name:<16}{label:<12}{delta:>+9.4f}{wp:>9.3f}{mp:>8.3f}  {mark}")
    print("-" * 78)
    print("delta = ablation - baseline.  'REJECT*' = survives Holm-Bonferroni at "
          f"alpha={alpha}\n(i.e. a genuine effect after multiple-comparison correction).")


def main():
    """
    For each ablation config it compares the per-seed values against the baseline
    on every reported metric using:
      - Welch's t-test (unequal variance; the right default at n=5),
      - Mann-Whitney U (rank-based, distribution-free fallback),
    and applies a Holm-Bonferroni correction across the whole family of tests so a
    single lucky comparison can't masquerade as a finding.
    """
    parser = argparse.ArgumentParser(
        description="Run significance tests (Welch t + MWU + Holm-Bonferroni) across ablation seeds vs baseline."
    )
    parser.add_argument(
        "--ablation-root", type=str,
        default=str(PathManager.TASK_1_DIR / "ablations"),
        help="Root directory containing ablation subdirectories"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Family-wise error rate for Holm-Bonferroni correction (default: 0.05)"
    )
    args = parser.parse_args()

    ablation_root = Path(args.ablation_root)
    if not ablation_root.exists():
        print(f"No ablation results at {ablation_root}.")
        raise SystemExit(1)

    print(f"Reading from: {ablation_root}\n")
    results = load_all_seeds(ablation_root)
    if not results:
        raise SystemExit(1)

    print_significance_summary(results, alpha=args.alpha)


if __name__ == "__main__":
    main()
