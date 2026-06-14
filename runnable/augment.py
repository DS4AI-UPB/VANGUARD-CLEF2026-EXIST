from exist_2026.dataset.augument_data import main
from exist_2026.path_manager import PathManager


def run_augment() -> None:
    """
    (Optional) Cross-lingual augmentation with NLLB-200 (EN<->ES).

    Roughly doubles the training set. Run after preprocess.py.

    Examples:
      python runnable/augment.py
      python runnable/augment.py --batch-size 32
      python runnable/augment.py --input-file="path-to-your-processed_data.json" --output-file="path-to-save-aug.json"
    """
    main(PathManager.DATA_EXIST_DIR)


if __name__ == "__main__":
    run_augment()
