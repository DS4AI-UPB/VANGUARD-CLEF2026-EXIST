import argparse
import sys
from pathlib import Path

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils

HIERARCHIES = {
    "2_2": {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []},
    "2_3": {
        "YES": [
            "IDEOLOGICAL-INEQUALITY",
            "STEREOTYPING-DOMINANCE",
            "OBJECTIFICATION",
            "SEXUAL-VIOLENCE",
            "MISOGYNY-NON-SEXUAL-VIOLENCE",
        ],
        "NO": [],
    },
}

METRICS = {
    "hard": {
        "2_1": ["ICM", "ICMNorm", "FMeasure"],
        "2_2": ["ICM", "ICMNorm", "FMeasure"],
        "2_3": ["ICM", "ICMNorm", "FMeasure"],
    },
    "soft": {
        "2_1": ["ICMSoft", "ICMSoftNorm", "CrossEntropy"],
        "2_2": ["ICMSoft", "ICMSoftNorm", "CrossEntropy"],
        "2_3": ["ICMSoft", "ICMSoftNorm"],
    },
}


def evaluate(pred_path: str, gold_path: str, task: str, mode: str) -> None:
    evaluator = PyEvALLEvaluation()
    params = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED}

    if task in HIERARCHIES:
        params[PyEvALLUtils.PARAM_HIERARCHY] = HIERARCHIES[task]

    metrics = METRICS[mode][task]

    print(f"\n{'=' * 60}")
    print(f"\tTask {task} | Mode: {mode}")
    print(f"\tPred: {pred_path}")
    print(f"\tGold: {gold_path}")
    print(f"\tMetrics: {metrics}")
    print(f"{'=' * 60}")

    report = evaluator.evaluate(pred_path, gold_path, metrics, **params)
    report.print_report()


def main():
    parser = argparse.ArgumentParser(description="EXIST 2026 PyEvALL Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred", type=str, help="Path to a single prediction file")
    group.add_argument("--pred-dir", type=str, help="Directory of prediction files (evaluates all)")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold standard file")
    parser.add_argument("--task", type=str, required=True, choices=["2_1", "2_2", "2_3"],
                        help="Subtask: 2_1, 2_2, or 2_3")
    parser.add_argument("--mode", type=str, required=True, choices=["hard", "soft"],
                        help="Evaluation mode: hard or soft")
    args = parser.parse_args()

    if args.pred:
        evaluate(args.pred, args.gold, args.task, args.mode)
    else:
        pred_dir = Path(args.pred_dir)
        pred_files = sorted(pred_dir.glob("*"))
        pred_files = [f for f in pred_files if f.is_file() and f.name != ".DS_Store"]

        if not pred_files:
            print(f"No prediction files found in {pred_dir}")
            sys.exit(1)

        print(f"Found {len(pred_files)} prediction files in {pred_dir}")
        for pred_file in pred_files:
            # Skip gold files if they're in the same directory
            if "gold" in pred_file.name.lower():
                continue
            try:
                evaluate(str(pred_file), args.gold, args.task, args.mode)
            except Exception as e:
                print(f"  ERROR evaluating {pred_file.name}: {e}")


if __name__ == "__main__":
    main()
