import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer, AutoImageProcessor

from exist_2026.consts.sensor_values import DefaultSensorValues
from exist_2026.dataset.extract_labels import extract_task_2_1_target, extract_task_2_2_target, extract_task_2_3_target

GENDER_MAP = {"M": 1, "F": 2, "Other": 3}
AGE_MAP = {"18-22": 1, "23-45": 2, "46+": 3}
STUDY_MAP = {
    "Less than high school diploma": 1,
    "High school degree or equivalent": 2,
    "Bachelor’s degree": 3,
    "Master’s degree": 4,
    "Doctorate": 5,
    "other": 6
}
ETH_MAP = {
    "White or Caucasian": 1,
    "Hispano or Latino": 2,
    "Black or African American": 3,
    "Asian": 4,
    "Multiracial": 5,
    "Middle Eastern": 6,
    "other": 7
}


def pad_to_square(img: Image.Image | torch.Tensor, fill: int = 0) -> torch.Tensor:
    """Pad a PIL image to a square with constant fill."""
    w, h = img.size
    max_wh = max(w, h)
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    return TF.pad(img, [hp, vp, max_wh - w - hp, max_wh - h - vp], fill, "constant")


def pad_list(lst: list, pad_val, length: int = 6):
    """Pad or truncate a list to a fixed length."""
    return lst[:length] + [pad_val] * max(0, length - len(lst))


class ExistMemeDataset(Dataset):
    """
    Dataset for EXIST 2026.

    Expects preprocessed JSON containing:
        - clean_text, clean_description: cleaned text files
        - processed_sensors: pre-computed [log_rt, fix, sac, hr_std] vector
        - meme: relative path to meme image
        - labels_task2_1: list of annotator labels (i.e., "YES" / "NO")
        - demographic fiels: gender_annotators, age_annotators, etc...
    """

    def __init__(
            self,
            json_path: str | Path,
            image_dir: str | Path,
            text_model_name: str,
            image_model_name: str,
            is_train: bool = False,
            scaler=None,
            tasks: set[str] | None = None
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
        self.is_train = is_train
        self.image_dir = image_dir
        self.scaler = scaler
        self.tasks = tasks or {"2.1", "2.2", "2.3"}

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_model_name, do_resize=False, do_center_crop=False
        )

        if self.is_train:
            self.augmentations = T.Compose([
                T.Resize((224, 224)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ])
        else:
            self.augmentations = T.Resize((224, 224))

        self.default_sensors = [
            float(np.log1p(DefaultSensorValues.REACTION_TIME)),
            DefaultSensorValues.FIXATIONS,
            DefaultSensorValues.SACCADES,
            DefaultSensorValues.HR_STD
        ]

        self.max_annotators = 6
        self.text_max_len = 128
        self.description_max_len = 512

    def _extract_optimized_sensors(self, item):
        """
        Extracts ONLY the statistically significant physiological features:
        [Reaction Time, Fixations Count, Saccades Count, HR Std Dev]
        Returns a 4D vector averaged across all users for the meme.
        """

        if "sensorial" not in item:
            return self.default_sensors

        users = item["sensorial"].get("users", [])
        if not users:
            return self.default_sensors

        modalities = item["sensorial"].get("modalities", {})
        et_data = modalities.get("ET", {}).get("by_user", {})
        hr_data = modalities.get("HR", {}).get("by_user", {})

        user_vectors = []
        for uid in users:
            u_et = et_data.get(uid, {})
            u_hr = hr_data.get(uid, {})

            rt = u_et.get("reaction_time")
            fc = u_et.get("fixations_count")
            sc = u_et.get("saccades_count")
            hrs = u_hr.get("garmin_hr_std")

            vec = [rt, fc, sc, hrs]
            vec = [v if v is not None and not np.isnan(v) else 0.0 for v in vec]
            user_vectors.append(vec)

        if not user_vectors:
            return self.default_sensors

        return np.mean(user_vectors, axis=0).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        meme_id = item.get("id_EXIST", str(idx))

        image_path = os.path.join(self.image_dir, item.get("meme", ""))
        image = self._load_image(image_path)
        image = pad_to_square(image, fill=0)
        image = self.augmentations(image)
        image_inputs = self.image_processor(images=image, return_tensors="pt")

        clean_text = item.get("clean_text", "")
        text_inputs = self.tokenizer(
            clean_text, padding="max_length", truncation=True, max_length=self.text_max_len, return_tensors="pt"
        )

        clean_desc = item.get("clean_description", "")
        desc_inputs = self.tokenizer(
            clean_desc, padding="max_length", truncation=True, max_length=self.description_max_len, return_tensors="pt"
        )

        genders = [GENDER_MAP.get(g, 3) for g in item.get("gender_annotators", [])]
        ages = [AGE_MAP.get(a, 2) for a in item.get("age_annotators", [])]
        studies = [STUDY_MAP.get(s, 6) for s in item.get("study_levels_annotators", [])]
        eths = [ETH_MAP.get(e, 7) for e in item.get("ethnicities_annotators", [])]

        # TODO: double check if extract_optimized_sensors() is the same as the preprocessing of the dataset
        # sensor_vector = self._extract_optimized_sensors(item)
        sensor_vector = item.get("processed_sensors", self.default_sensors)
        if self.scaler is not None:
            sensor_vector = self.scaler.transform([sensor_vector])[0].tolist()

        sample = {
            "id": meme_id,
            "text_input_ids": text_inputs["input_ids"].squeeze(0),
            "text_attention_mask": text_inputs["attention_mask"].squeeze(0),
            "desc_input_ids": desc_inputs["input_ids"].squeeze(0),
            "desc_attention_mask": desc_inputs["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "genders": torch.tensor(pad_list(genders, 0), dtype=torch.long),
            "ages": torch.tensor(pad_list(ages, 0), dtype=torch.long),
            "studies": torch.tensor(pad_list(studies, 0), dtype=torch.long),
            "eths": torch.tensor(pad_list(eths, 0), dtype=torch.long),
            "sensorial": torch.tensor(sensor_vector, dtype=torch.float32),
        }

        if "2.1" in self.tasks:
            sample["target_2_1"] = extract_task_2_1_target(item)
        if "2.2" in self.tasks:
            sample["target_2_2"] = extract_task_2_2_target(item)
        if "2.3" in self.tasks:
            sample["target_2_3"] = extract_task_2_3_target(item)
        return sample

    @staticmethod
    def _load_image(image_path: Path | str) -> Image.Image:
        try:
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img = img.convert("RGBA")
                    background = Image.new("RGB", img.size, (0, 0, 0))
                    background.paste(img, mask=img.split()[3])
                    return background
                return img.convert("RGB")
        except Exception:
            return Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
