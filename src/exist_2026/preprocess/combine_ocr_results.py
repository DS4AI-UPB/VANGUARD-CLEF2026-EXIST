import argparse
import json
from pathlib import Path


def combine_ocr_results_with_fixes(main_path: str | Path, fixed_path: str | Path, save_path: str | Path) -> None:
    with open(main_path, 'r', encoding='utf-8') as f:
        main = json.load(f)

    with open(fixed_path, 'r', encoding='utf-8') as f:
        fixed = json.load(f)

    main.update(fixed)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(main, f, ensure_ascii=False, indent=4)

    print(f"Merged {len(fixed)} fixed entries into main OCR file. Total: {len(main)}")


def main():
    """
    Use:
        ocr_results.json
        ocr_results_corrected.json
        ocr_results_combined.json
    For each train or test configuration.
    """
    parser = argparse.ArgumentParser(description="Merge corrected OCR entries back into the main OCR results file.")
    parser.add_argument("--main-file", type=str, required=True, help="Path to the main OCR results file")
    parser.add_argument("--fixed-file", type=str, required=True, help="Path to the corrected entries file")
    parser.add_argument("--save-file", type=str, required=True, help="Path to save the merged output")
    args = parser.parse_args()

    combine_ocr_results_with_fixes(args.main_file, args.fixed_file, args.save_file)


if __name__ == "__main__":
    main()
