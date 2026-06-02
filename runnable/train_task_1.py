from exist_2026.path_manager import PathManager
from exist_2026.train.nn.determinism import seed_everything
from exist_2026.train.train_t1 import train_and_validate_model


def main():
    seed_everything(seed=42)
    train_and_validate_model(
        json_path=PathManager.DATA_DIR / "processed_data_augmented.json",
        img_dir=PathManager.DATA_DIR / "memes",
        test_json=PathManager.DATA_DIR / "test" / "processed_data.json",
        test_img_dir=PathManager.DATA_DIR / "test" / "memes",
        save_dir=PathManager.TASK_1_DIR
    )


if __name__ == "__main__":
    main()
