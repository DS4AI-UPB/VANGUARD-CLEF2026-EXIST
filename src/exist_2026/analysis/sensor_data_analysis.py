import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def parse_dataset(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations_list = []
    instances_list = []

    for item in data:
        meme_id = item.get("id_EXIST")

        num_annotators = item.get("number_annotators", 0)
        labels = item.get("labels_task2_1", [])

        yes_count = 0
        for i in range(num_annotators):
            label = 1 if labels[i] == "YES" else 0
            yes_count += label
            annotations_list.append({
                "meme_id": meme_id,
                "annotator_id": item["annotators"][i],
                "gender": item["gender_annotators"][i],
                "age": item["age_annotators"][i],
                "study_level": item["study_levels_annotators"][i],
                "ethnicity": item["ethnicities_annotators"][i],
                "label": label
            })

        consensus = 1 if (yes_count / num_annotators) >= 0.5 else 0
        yes_ratio = yes_count / num_annotators

        sensors = item.get("sensorial", {}).get("modalities", {})
        et_data = sensors.get("ET", {}).get("by_user", {})
        hr_data = sensors.get("HR", {}).get("by_user", {})
        eeg_data = sensors.get("EEG", {}).get("by_user", {})

        user_metrics = []
        all_users = set(et_data.keys()).union(hr_data.keys()).union(eeg_data.keys())

        for user in all_users:
            metrics = {}
            if user in et_data:
                u_et = et_data[user]
                metrics["reaction_time"] = u_et.get("reaction_time", np.nan)
                metrics["fixations_count"] = u_et.get("fixations_count", np.nan)
                metrics["saccades_count"] = u_et.get("saccades_count", np.nan)
                metrics["pupil_left_mean"] = u_et.get("3d_eye_states_pupil diameter left [mm]_mean", np.nan)

            if user in hr_data:
                u_hr = hr_data[user]
                metrics["hr_mean"] = u_hr.get("garmin_hr_mean", np.nan)
                metrics["hr_std"] = u_hr.get("garmin_hr_std", np.nan)

            if user in eeg_data:
                u_eeg = eeg_data[user]
                for band in ["Alpha", "Beta", "Gamma", "Theta", "Delta"]:
                    band_vals = [v for k, v in u_eeg.items() if f"{band}_power" in k]
                    if band_vals:
                        metrics[f"eeg_{band.lower()}"] = np.nanmean(band_vals)

            user_metrics.append(metrics)

        if user_metrics:
            df_temp = pd.DataFrame(user_metrics)
            agg_metrics = df_temp.mean(skipna=True).to_dict()
        else:
            agg_metrics = {}

        instance_data = {
            "meme_id": meme_id,
            "consensus": consensus,
            "yes_ratio": yes_ratio
        }
        instance_data.update(agg_metrics)
        instances_list.append(instance_data)

    df_annot = pd.DataFrame(annotations_list)
    df_inst = pd.DataFrame(instances_list)

    df_inst["consensus"] = df_inst["consensus"].astype(int)

    return df_annot, df_inst


def generate_statistical_tables(df_annot, df_inst):
    """
    Computes Chi-Square for demographics and T-tests for physiological data,
    printing the results to mimic Table 1.
    """
    print("-" * 60)
    print("Table 1: Statistical Significance of Features")
    print("-" * 60)

    print("Demographics (Chi-Square Tests vs Labeling Behavior)")
    demo_vars = ["gender", "age", "study_level"]
    for var in demo_vars:
        contingency = pd.crosstab(df_annot[var], df_annot["label"])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        sig = "**" if p < 0.05 else ""

        p_str = "< 0.001" if p < 0.001 else f"= {p:.4f}"
        print(f" - {var.capitalize():<15}: chi2 = {chi2:>6.2f}, p-value {p_str:>7} {sig}")

    print("\nPhysiological (Independent T-Tests: Consensus 0 vs 1)")
    group0 = df_inst[df_inst["consensus"] == 0]
    group1 = df_inst[df_inst["consensus"] == 1]

    physio_vars = {
        "reaction_time": "Reaction Time",
        "fixations_count": "Fixations Count",
        "saccades_count": "Saccades Count",
        "hr_std": "HR Std.",
        "pupil_left_mean": "Mean Pupil Diameter",
        "hr_mean": "Mean Heart Rate",
        "eeg_alpha": "EEG Alpha Power"
    }

    for col, label in physio_vars.items():
        if col in df_inst.columns:
            g0_clean = group0[col].dropna()
            g1_clean = group1[col].dropna()

            if len(g0_clean) > 0 and len(g1_clean) > 0:
                t_stat, p_val = stats.ttest_ind(g1_clean, g0_clean, equal_var=True)

                sig = "**" if p_val < 0.05 else ""

                p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
                print(f" - {label:<20}: t = {t_stat:>6.2f}, p-value {p_str:<8} {sig}")

    print("-" * 60)


def plot_relevant_features_boxplots(df_inst):
    sns.set_theme(style="white")

    features_to_plot = [
        ("reaction_time", "reaction_time"),
        ("fixations_count", "fixations_count"),
        ("saccades_count", "saccades_count"),
        ("hr_std", "hr_std")
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    palette = {0: "#74c4a4", 1: "#f49c7c"}

    for i, (col, title) in enumerate(features_to_plot):
        if col in df_inst.columns:
            sns.boxplot(
                data=df_inst,
                x="consensus",
                y=col,
                hue="consensus",
                legend=False,
                ax=axes[i],
                palette=palette,
                linewidth=1,
                fliersize=5
            )
            axes[i].set_title(f"{title} by Meme Label")
            axes[i].set_xlabel("Consensus: Sexist? (0=No, 1=Yes)")
            axes[i].set_ylabel(title)

    plt.tight_layout()
    plt.savefig("relevant_features_boxplots.png", dpi=300)
    plt.show()


def plot_correlation_heatmap(df_inst):
    cols_order = [
        "reaction_time", "fixations_count", "saccades_count",
        "pupil_left_mean", "hr_mean", "hr_std",
        "eeg_alpha", "eeg_beta", "eeg_gamma", "eeg_theta", "eeg_delta",
        "yes_ratio"
    ]

    avail_cols = [c for c in cols_order if c in df_inst.columns]
    df_corr = df_inst[avail_cols].corr(method="spearman")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-0.1,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": .9}
    )
    plt.title("Spearman Correlation: Physiological Sensors & Sexism Ratio")
    plt.tight_layout()
    plt.savefig("sensor_correlation_heatmap.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    from exist_2026.path_manager import PathManager

    df_annot, df_inst = parse_dataset(PathManager.DATA_DIR / "processed_data.json")

    generate_statistical_tables(df_annot, df_inst)
    plot_relevant_features_boxplots(df_inst)
    plot_correlation_heatmap(df_inst)