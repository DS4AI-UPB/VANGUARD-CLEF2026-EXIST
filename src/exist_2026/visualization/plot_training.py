from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_results(csv_path: str | Path, save_dir: str | Path | None = None) -> None:
    df = pd.read_csv(csv_path)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    ax.plot(df["Epoch"], df["Train_Loss"], label="Train", marker="o", alpha=0.7)
    ax.plot(df["Epoch"], df["Val_Loss"], label="Val", marker="s", color="crimson", lw=2.5)
    best_idx = df["Val_Loss"].idxmin()
    ax.annotate(
        f"Best: Epoch {df.iloc[best_idx]['Epoch']}",
        xy=(df.iloc[best_idx]["Epoch"], df.iloc[best_idx]["Val_Loss"]),
        xytext=(10, 10), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.set_title("Loss (Lower is Better)")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["Epoch"], df["Train_Acc"], label="Train", marker="o", color="forestgreen", alpha=0.7)
    ax.plot(df["Epoch"], df["Val_Acc"], label="Val", marker="s", color="lime", lw=2.5)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df["Epoch"], df["Train_F1_Macro"], label="Train", marker="o", color="darkorange", alpha=0.7)
    ax.plot(df["Epoch"], df["Val_F1_Macro"], label="Val", marker="*", color="gold", ms=12, lw=2.5)
    best_f1_idx = df["Val_F1_Macro"].idxmax()
    ax.axvline(df.iloc[best_f1_idx]["Epoch"], color="grey", ls="--", alpha=0.5)
    ax.set_title("Macro F1-Score (Primary Metric)")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 1]
    f1_gap = df["Train_F1_Macro"] - df["Val_F1_Macro"]
    ax.fill_between(df["Epoch"], f1_gap, color="orange", alpha=0.3, label="F1 Gap")
    ax.plot(df["Epoch"], f1_gap, color="darkorange", marker="x")
    ax.axhline(0.1, color="red", ls="--", label="Danger Zone (>0.1)")
    ax.set_title("Overfitting Monitor")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train - Val")
    ax.legend()

    for ax in axes.flat:
        ax.set_xticks(df["Epoch"].astype(int))

    best_val_f1 = df["Val_F1_Macro"].max()
    best_epoch = df.iloc[best_f1_idx]["Epoch"]
    fig.suptitle(
        f"EXIST 2026 Task 2.1 - Training (Best Val F1: {best_val_f1:.4f} at Epoch {best_epoch:.0f})",
        fontsize=16,
    )
    if save_dir is not None:
        out_path = Path(save_dir) / "training_curves.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    from exist_2026.path_manager import PathManager

    plot_training_results(PathManager.BASE_DIR / "training_log.csv", PathManager.BASE_DIR)
