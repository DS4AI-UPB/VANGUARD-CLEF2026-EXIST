import argparse
import json
import os
from pathlib import Path

from exist_2026.path_manager import PathManager


def process_and_validate_ocr(
        main_json_path: str | Path, ocr_json_path: str | Path, error_output_path: str | Path
) -> None:
    """
    Cleans ocr_results_v2 by moving parsing errors to a separate file,
    then validates the remaining data against the training set.
    """

    if not os.path.exists(main_json_path):
        print(f"Error: {main_json_path} not found.")
        return
    with open(main_json_path, 'r', encoding='utf-8') as f:
        main_raw = json.load(f)

    main_data = list(main_raw.values()) if isinstance(main_raw, dict) else main_raw

    if not os.path.exists(ocr_json_path):
        print(f"Error: {ocr_json_path} not found.")
        return
    with open(ocr_json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    parsed_errors = {}
    ids_to_remove = []

    for meme_id, entry in ocr_data.items():
        if entry.get("text") == "ERROR_PARSING_JSON":
            parsed_errors[meme_id] = entry
            ids_to_remove.append(meme_id)

    if ids_to_remove:
        print(f"Found {len(ids_to_remove)} parsing errors. Extracting...")

        for meme_id in ids_to_remove:
            del ocr_data[meme_id]

        with open(error_output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_errors, f, indent=4, ensure_ascii=False)

        with open(ocr_json_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_data, f, indent=4, ensure_ascii=False)

        print(f"✅ Extracted errors saved to: {error_output_path}")
        print(f"✅ Cleaned OCR file updated: {ocr_json_path}")
    else:
        print("✨ No 'ERROR_PARSING_JSON' entries found. No cleanup needed.")

    issues = {
        "missing_id_EXIST": [],
        "empty_main_text": [],
        "missing_in_ocr": [],
        "empty_ocr_text": []
    }

    print(f"\n--- Starting Validation on {len(main_data)} items ---")

    for i, item in enumerate(main_data):
        meme_id = item.get("id_EXIST")
        if not meme_id:
            issues["missing_id_EXIST"].append(f"Index {i}")
            continue

        meme_id_str = str(meme_id)

        if not item.get("text") or str(item.get("text")).strip() == "":
            issues["empty_main_text"].append(meme_id_str)

        if meme_id_str not in ocr_data:
            issues["missing_in_ocr"].append(meme_id_str)
        else:
            ocr_entry = ocr_data[meme_id_str]
            if not ocr_entry.get("text") or str(ocr_entry.get("text")).strip() == "":
                issues["empty_ocr_text"].append(meme_id_str)

    print("\n--- Validation Report ---")

    if not any(issues.values()):
        print("✅ No issues found! All IDs match and text fields are populated.")
    else:
        for category, list_of_ids in issues.items():
            if list_of_ids:
                print(f"❌ {category.replace('_', ' ').title()}: {len(list_of_ids)} found")
                print(f"   Examples: {', '.join(list_of_ids[:8])}{'...' if len(list_of_ids) > 8 else ''}")
            else:
                print(f"✅ {category.replace('_', ' ').title()}: None")


def main(default_data_dir: str | Path):
    default_data_dir = Path(default_data_dir)
    parser = argparse.ArgumentParser(description="Clean and validate OCR generated JSON against the main JSON file")
    parser.add_argument(
        "--main-file", type=str, default=str(default_data_dir / "EXIST2026_test_clean"),
        help="Path to the main (gold standard) JSON file"
    )
    parser.add_argument(
        "--ocr-file", type=str, default=str(default_data_dir / "ocr_results.json"),
        help="Path to the OCR results JSON file"
    )
    parser.add_argument(
        "--error-file", type=str, default=str(default_data_dir / "ocr_parsing_errors.json"),
        help="Path to output the parsing errors JSON file"
    )
    args = parser.parse_args()

    process_and_validate_ocr(args.main_file, args.ocr_file, args.error_file)


if __name__ == "__main__":
    # default_dir = PathManager.DATA_EXIST_DIR / "training"
    # MAIN_FILE = DATA_FILE / "EXIST2026_training.json"
    # MAIN_FILE = DATA_FILE / "EXIST2026_test_clean.json"

    default_dir = PathManager.DATA_EXIST_DIR / "test"
    main(default_dir)
