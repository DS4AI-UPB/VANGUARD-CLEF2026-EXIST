import json
import tempfile


def _write_tmp_json(data: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _extract_metrics(report_json: dict) -> dict[str, float]:
    """Pull metric averages out of the PyEvALL embedded report dict."""
    results = {}
    for metric_key, metric_val in report_json.get("metrics", {}).items():
        if metric_val.get("status") != "OK":
            continue
        avg = metric_val.get("results", {}).get("average_per_test_case")
        if avg is not None:
            results[metric_key] = float(avg)
        # For FMeasure, also grab percalss scores
        for tc in metric_val.get("results", {}).get("test_cases", []):
            if "classes" in tc:
                for cls_name, cls_val in tc["classes"].items():
                    results[f"{metric_key}_{cls_name}"] = float(cls_val)
    return results
