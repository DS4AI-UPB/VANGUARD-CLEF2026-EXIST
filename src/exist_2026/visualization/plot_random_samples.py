import random
import textwrap
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


SENSOR_LABELS = ("HR (bpm)", "Pupil L (mm)", "Fixations", "EEG α", "EEG β")


def plot_random_samples(
    dataset,
    n_samples: int = 5,
    seed: Optional[int] = None,
    figsize: Optional[tuple] = None,
    suptitle: str = "EXIST memes — random batch",
):
    n_samples = min(n_samples, len(dataset))
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=n_samples)

    figsize = figsize or (12, 2.9 * n_samples)
    fig, axes = plt.subplots(
        n_samples, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )
    if n_samples == 1:
        axes = np.array([axes])

    mean = np.array(dataset.image_processor.image_mean)
    std = np.array(dataset.image_processor.image_std)

    for row, idx in enumerate(indices):
        sample = dataset[idx]

        img = sample["pixel_values"]
        img = img.to("cpu").numpy()
        img = img.transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        ax_img = axes[row, 0]
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        p_yes = float(sample["label"])
        lang = sample.get("lang", "?")
        img_name = sample.get("image_name", "?")
        n_ann = sample.get("n_annotators", 0)
        n_yes = round(p_yes * n_ann) if n_ann else 0

        ax_img.set_title(
            f"[{lang}] {img_name}   p_yes = {p_yes:.2f}  ({n_yes}/{n_ann} YES)",
            fontsize=9,
        )

        text = sample.get("raw_text", "") or "(no text)"
        wrapped = "\n".join(textwrap.wrap(text, width=55))[:300]
        ax_img.set_xlabel(wrapped, fontsize=8)

        # color the frame by agreement (red = high sexist agreement)
        for spine in ax_img.spines.values():
            spine.set_edgecolor(plt.cm.RdYlGn_r(p_yes))
            spine.set_linewidth(2.5)

        ax_sen = axes[row, 1]
        vals = sample["sensor_feat"].numpy()
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(vals)))

        y_pos = np.arange(len(vals))
        ax_sen.barh(y_pos, vals, color=colors)
        ax_sen.set_yticks(y_pos)
        ax_sen.set_yticklabels(SENSOR_LABELS, fontsize=8)
        ax_sen.axvline(0, color="grey", lw=0.6)
        ax_sen.set_title("Sensorial features (mean across users)", fontsize=9)
        ax_sen.invert_yaxis()
        ax_sen.tick_params(axis="x", labelsize=8)

        for i, v in enumerate(vals):
            ax_sen.text(
                v, i, f" {v:.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


if __name__ == "__main__":
    from exist_2026.dataset.data_loader import ExistMemeDataset

    dataset = ExistMemeDataset(
        json_path="../../../data/EXIST2026_training.json",
        image_dir="../../../data/memes",
        text_model_name="distilbert-base-multilingual-cased",
        image_model_name="google/vit-base-patch16-224-in21k",
    )

    fig = plot_random_samples(dataset, n_samples=5)
    fig.savefig("sample_batch.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved sample_batch.png")