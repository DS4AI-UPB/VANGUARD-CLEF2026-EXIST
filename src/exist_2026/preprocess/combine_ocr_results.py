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


if __name__ == '__main__':
    main_p = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/ocr_results.json"
    fixed_p = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/ocr_results_corrected.json"
    save_p = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/ocr_results_combined.json"
    combine_ocr_results_with_fixes(main_p, fixed_p, save_p)

    main_p = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/ocr_results.json"
    fixed_p = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/ocr_results_corrected.json"
    save_p = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/ocr_results_combined.json"
    combine_ocr_results_with_fixes(main_p, fixed_p, save_p)
