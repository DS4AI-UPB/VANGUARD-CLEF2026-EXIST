import argparse
from pathlib import Path

from exist_2026.preprocess.extract_ocr import improve_meme_ocr
from exist_2026.preprocess.preprocessing import preprocess_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data", type=Path, default=Path("data/exist-memes"),
        help="Dataset root containing training/ and test/."
    )
    p.add_argument(
        "--splits", nargs="+", default=["training", "test"], choices=["training", "test"],
        help="Which splits to preprocess."
    )
    p.add_argument("--train-raw", default="EXIST2026_training.json", help="Raw training JSON filename.")
    p.add_argument("--test-raw", default="EXIST2026_test_clean.json", help="Raw test JSON filename.")
    return p.parse_args()


def main() -> None:
    """
    Preprocess the VANGUARD EXIST 2026 data: VLM OCR extraction + cleaning.

      - pip install -e .        (so `exist_2026` is importable)
      - raw data placed under --data
      - a local Ollama server with the model pulled:  ollama pull gemma4:e4b

    Examples:
      python runnable/preprocess.py
      python runnable/preprocess.py --splits training
      python runnable/preprocess.py --data /path/to/exist-memes
    """
    args = parse_args()
    raw_name = {"training": args.train_raw, "test": args.test_raw}

    for split in args.splits:
        d = args.data / split
        raw = d / raw_name[split]
        ocr = d / "ocr_results.json"
        out = d / "processed_data.json"

        print(f"\n=== {split}: OCR + visual-description extraction ===")
        improve_meme_ocr(image_dir=d / "memes", metadata_path=raw, output_file=ocr)

        print(f"\n=== {split}: cleaning + sensor features ===")
        preprocess_data(input_json_path=raw, ocr_json_path=ocr, output_json_path=out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
