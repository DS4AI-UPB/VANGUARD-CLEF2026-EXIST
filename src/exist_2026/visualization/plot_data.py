import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# TODO: needs a massive refactor
def analyze_dataset(json_path="data/EXIST2026_training.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data

    extracted = []
    for item in data:
        labels = item.get("labels_task2_1", [])
        yes_count = sum(1 for label in labels if label.upper() == "YES")
        no_count = len(labels) - yes_count

        soft_label = yes_count / len(labels) if len(labels) > 0 else 0.0

        hard_label = "Sexist" if soft_label > 0.5 else "Non-Sexist"

        lang = item.get("lang", "Unknown")

        hr_data = item.get("sensorial", {}).get("modalities", {}).get("HR", {}).get("by_user", {})
        hr_values = [v["garmin_hr_mean"] for v in hr_data.values() if v.get("garmin_hr_mean")]
        avg_hr = np.mean(hr_values) if hr_values else None

        extracted.append({
            "Soft_Label": soft_label,
            "Hard_Label": hard_label,
            "Language": lang,
            "Avg_Heart_Rate": avg_hr
        })

    df = pd.DataFrame(extracted)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    class_counts = df["Hard_Label"].value_counts()
    axes[0].pie(class_counts, labels=class_counts.index, autopct="%1.1f%%",
                colors=["#ff9999", "#66b3ff"], startangle=140, explode=(0.05, 0))
    axes[0].set_title("Dataset Composition (Majority Vote)", fontsize=14, fontweight="bold")

    sns.histplot(df["Soft_Label"], bins=11, kde=True, ax=axes[1], color="royalblue")
    axes[1].set_title("Distribution of Annotator Agreement", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Proportion of 'YES' votes")

    sns.countplot(data=df, x="Language", hue="Hard_Label", ax=axes[2], palette="pastel")
    axes[2].set_title("Sexist Content by Language", fontsize=14, fontweight="bold")

    df_hr = df.dropna(subset=["Avg_Heart_Rate"])
    sns.regplot(data=df_hr, x="Soft_Label", y="Avg_Heart_Rate", scatter_kws={"alpha": 0.3},
                line_kws={"color": "red"}, ax=axes[3])
    axes[3].set_title("Viewer Heart Rate vs Sexism Score", fontsize=14, fontweight="bold")
    axes[3].set_xlabel("Sexism Score (0 = None, 1 = Full Agreement)")

    plt.suptitle(f"EXIST 2026 Dataset Analysis - {len(df)} Total Memes", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig("explicit_analysis.png", dpi=300, bbox_inches="tight")
    print("Explicit analysis saved to 'explicit_analysis.png'")
    plt.show()


if __name__ == "__main__":
    analyze_dataset()