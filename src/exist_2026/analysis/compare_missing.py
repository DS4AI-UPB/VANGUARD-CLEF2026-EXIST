import json
import os
from pathlib import Path


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


if __name__ == "__main__":
    DATA_FILE = Path("/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test")
    MAIN_FILE = DATA_FILE / "EXIST2026_test_clean.json"
    # DATA_FILE = Path("/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training")
    # MAIN_FILE = DATA_FILE / "EXIST2026_training.json"
    OCR_FILE = DATA_FILE / "ocr_results.json"
    ERROR_FILE = DATA_FILE / "ocr_parsing_errors.json"

    process_and_validate_ocr(MAIN_FILE, OCR_FILE, ERROR_FILE)
