import os
import json
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
import string
from exist_2026.path_manager import PathManager

JSON_DIR = PathManager.DATA_DIR / "og"
PLOTS_DIR = PathManager.ANALYSIS_DIR / "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(json_dir):
    """Loads JSONs and tracks the source filename for each record."""
    data_records = []
    json_files = glob.glob(os.path.join(json_dir, "EXIST2026_training.json"))

    for file_path in json_files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
                for record_id, record_data in content.items():
                    record_data["source_json"] = filename
                    data_records.append(record_data)
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

    return pd.DataFrame(data_records)


def extract_image_features(df, base_dir):
    """Extracts width, height, aspect ratio, and average brightness using the dataset"s path."""
    widths, heights, aspect_ratios, brightness = [], [], [], []

    for _, row in df.iterrows():
        relative_path = row.get("path_memes", "")
        img_path = os.path.join(base_dir, relative_path) if relative_path else ""

        if img_path and os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)

                    gray_img = img.convert("L")
                    stat = np.mean(np.array(gray_img))
                    brightness.append(stat)
            except Exception:
                widths.append(np.nan)
                heights.append(np.nan)
                aspect_ratios.append(np.nan)
                brightness.append(np.nan)
        else:
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)
            brightness.append(np.nan)

    df["img_width"] = widths
    df["img_height"] = heights
    df["img_aspect_ratio"] = aspect_ratios
    df["img_brightness"] = brightness
    return df


def analyze_and_plot(df):
    logger.info(f"Loaded {len(df)} records. Starting analysis...")

    df["char_count"] = df["text"].astype(str).apply(len)
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    logger.info("========== DATASET STATISTICS ==========")

    text_stats = df[["char_count", "word_count"]].describe().round(2)
    logger.info(f"\n--- Text Analysis (Lengths, Mean, Std) ---\n{text_stats.to_string()}")

    lang_counts = df["lang"].value_counts()
    logger.info(f"\n--- Language Distribution ---\n{lang_counts.to_string()}")

    if "img_brightness" in df.columns and not df["img_brightness"].isna().all():
        img_stats = df[["img_width", "img_height", "img_aspect_ratio", "img_brightness"]].describe().round(2)
        logger.info(f"\n--- Image Analysis (Dimensions, Ratios, Brightness) ---\n{img_stats.to_string()}")

    logger.info("========================================")

    logger.info("Generating and saving plots...")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=lang_counts.index, y=lang_counts.values, palette="viridis")
    plt.title("Distribution of Languages")
    plt.xlabel("Language")
    plt.ylabel("Number of Memes")
    plt.savefig(os.path.join(PLOTS_DIR, "1_language_distribution.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["char_count"], bins=30, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Distribution of Character Counts")
    axes[0].set_xlabel("Number of Characters")

    sns.histplot(df["word_count"], bins=30, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Distribution of Word Counts")
    axes[1].set_xlabel("Number of Words")
    plt.savefig(os.path.join(PLOTS_DIR, "2_text_length_distributions.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

    all_text = " ".join(df["text"].astype(str).tolist())
    all_text = all_text.translate(str.maketrans("", "", string.punctuation)).lower()

    if all_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Frequent Words Across All Memes")
        plt.savefig(os.path.join(PLOTS_DIR, "3_text_wordcloud.pdf"), format="pdf", bbox_inches="tight")
        plt.close()

    task1_labels = [label for sublist in df["labels_task2_1"].dropna() for label in sublist]
    task2_labels = [label for sublist in df["labels_task2_2"].dropna() for label in sublist]
    task3_labels = [label for sublist in df["labels_task2_3"].dropna() for inner_list in sublist for label in
                    inner_list]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pd.Series(task1_labels).value_counts().plot(kind="bar", ax=axes[0], color="cornflowerblue")
    axes[0].set_title("Task 1 Labels (Sexism Existence)")
    axes[0].tick_params(axis="x", rotation=45)

    pd.Series(task2_labels).value_counts().plot(kind="bar", ax=axes[1], color="mediumseagreen")
    axes[1].set_title("Task 2 Labels (Direct vs Indirect)")
    axes[1].tick_params(axis="x", rotation=45)

    pd.Series(task3_labels).value_counts().head(10).plot(kind="bar", ax=axes[2], color="indianred")
    axes[2].set_title("Top 10 Task 3 Labels (Subtypes)")
    axes[2].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "4_annotation_label_distributions.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

    if "img_width" in df.columns and not df["img_width"].isna().all():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.scatterplot(x="img_width", y="img_height", data=df, ax=axes[0], alpha=0.5, color="purple", edgecolor=None)
        axes[0].set_title("Image Dimensions (Width vs. Height)")
        axes[0].set_xlabel("Width (pixels)")
        axes[0].set_ylabel("Height (pixels)")

        sns.histplot(df["img_aspect_ratio"], bins=30, kde=True, ax=axes[1], color="teal")
        axes[1].set_title("Distribution of Image Aspect Ratios")
        axes[1].set_xlabel("Aspect Ratio (Width/Height)")
        axes[1].axvline(1.0, color="red", linestyle="--", label="1:1 Square")
        axes[1].legend()

        plt.savefig(os.path.join(PLOTS_DIR, "5_image_dimensions.pdf"), format="pdf", bbox_inches="tight")
        plt.close()

    if "img_brightness" in df.columns and not df["img_brightness"].isna().all():
        plt.figure(figsize=(8, 5))
        sns.histplot(df["img_brightness"], bins=30, kde=True, color="orange")
        plt.title("Distribution of Image Brightness (0=Black, 255=White)")
        plt.xlabel("Average Pixel Brightness")
        plt.ylabel("Count")
        plt.savefig(os.path.join(PLOTS_DIR, "6_image_brightness.pdf"), format="pdf", bbox_inches="tight")
        plt.close()

    logger.info(f"Analysis complete! All publication-ready PDFs have been saved to the '{PLOTS_DIR}' directory.")


if __name__ == "__main__":
    logger.info("Initializing dataset load...")
    df = load_dataset(JSON_DIR)

    if not df.empty:
        logger.info("Extracting image features using paths from the dataset...")
        df = extract_image_features(df, PathManager.DATA_DIR)
        analyze_and_plot(df)
    else:
        logger.warning(f"No data found. Please check your '{JSON_DIR}' directory for JSON files.")