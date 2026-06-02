from pathlib import Path
from typing import Final


class PathManager:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent

    DATA_DIR: Final[Path] = BASE_DIR / "data"
    DATA_EXIST_DIR: Final[Path] = DATA_DIR / "exist-memes"
    OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
    ANALYSIS_DIR: Final[Path] = OUTPUT_DIR / "analysis"

    RESULTS_DIR: Final[Path] = OUTPUT_DIR / "results"
    TASK_1_DIR: Final[Path] = RESULTS_DIR / "task_1"
    TASK_2_DIR: Final[Path] = RESULTS_DIR / "task_2"
    TASK_3_DIR: Final[Path] = RESULTS_DIR / "task_3"

    SENSOR_WEIGHTS = BASE_DIR / "smart_sensor_weights.pt"


if __name__ == "__main__":
    print(Path(__file__).resolve().parent)
    print(PathManager.BASE_DIR)
