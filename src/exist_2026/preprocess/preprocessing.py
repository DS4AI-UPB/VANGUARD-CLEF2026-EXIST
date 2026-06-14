import argparse
import json
import re
import unicodedata
from pathlib import Path

import emoji
import numpy as np

from exist_2026.consts.sensor_values import DefaultSensorValues


def clean_meme_text_for_transformer(text: str) -> str:
    """
    Clean meme text for transformer input.

    Preserves casing to capture sarcastic/ironic "Spongebob" casing.
    Removes URLs, HTML tags, social media UI artifacts etc.
    Normalizes punctuation while keeping bilingual character support (EN/ES).
    """
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    ui_patterns = [
        r"(?i)\d+[\s,.]*(me gusta|comentarios|likes|shares|recomendar|enviar)",
        r"(?i)me gusta comentar compartir"
    ]
    for pattern in ui_patterns:
        text = re.sub(pattern, "", text)

    text = re.sub(r"([!¡?¿.])\s+", r"\1", text)
    text = re.sub(r"([!¡?¿.])\1+", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!¡?¿áéíóúñÁÉÍÓÚÑ:_]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_sensors_for_dl(item: dict[str, dict[str, float]]) -> list[float]:
    """
    Pre-computes the 4-feature Tabular array representing 4 statistically significant features:
        [Log_Reaction_Time, Mean_Fixations, Mean_Saccades, Mean_HR_Std]

    Values are averaged across all annotator users for the meme.
    Reaction time is log-transformed to compress its large range.
    """
    sensorial_data = item.get("sensorial", {}).get("modalities", {})
    rt_list, fix_list, sac_list, hr_list = [], [], [], []

    for _user, data in sensorial_data.get("ET", {}).get("by_user", {}).items():
        rt = data.get("reaction_time")
        fix = data.get("fixations_count")
        sac = data.get("saccades_count")

        rt_list.append(rt if rt is not None else DefaultSensorValues.REACTION_TIME)
        fix_list.append(fix if fix is not None else DefaultSensorValues.FIXATIONS)
        sac_list.append(sac if sac is not None else DefaultSensorValues.SACCADES)

    for _user, data in item.get("sensorial", {}).get("HR", {}).get("by_user", {}).items():
        hrs = data.get("garmin_hr_std")
        hr_list.append(hrs if hrs is not None else DefaultSensorValues.HR_STD)

    avg_rt = float(np.log1p(np.mean(rt_list))) if rt_list else float(np.log1p(DefaultSensorValues.REACTION_TIME))
    avg_fix = float(np.mean(fix_list)) if fix_list else DefaultSensorValues.FIXATIONS
    avg_sac = float(np.mean(sac_list)) if sac_list else DefaultSensorValues.SACCADES
    avg_hr = float(np.mean(hr_list)) if hr_list else DefaultSensorValues.HR_STD

    return [avg_rt, avg_fix, avg_sac, avg_hr]


def preprocess_data(input_json_path: str | Path, ocr_json_path: str | Path, output_json_path: str | Path) -> None:
    """
    Full preprocessing pipeline:
        1. Load raw dataset and ocr results
        2. Merge ocr text with original meme text
        3. Clean text and description for transformer input
        4. Pre-compute sensor feature vectors
        5. Save procesed dataset
    """
    input_json_path = Path(input_json_path)
    ocr_json_path = Path(ocr_json_path)
    output_json_path = Path(output_json_path)

    if not input_json_path.exists():
        print(f"Error: {input_json_path} not found.")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        data_items = raw_data.items()
    else:
        data_items = [(item.get("id_EXIST", idx), item) for idx, item in enumerate(raw_data)]

    ocr_dict = {}
    if ocr_json_path.exists():
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            ocr_dict = json.load(f)
    else:
        print(f"Warning: {ocr_json_path} not found. Proceeding with original text only.")

    processed_list = []

    for meme_id, item in data_items:
        meme_id_str = str(meme_id)
        ocr_entry = ocr_dict.get(meme_id_str, "")

        if isinstance(ocr_entry, dict):
            ocr_text = ocr_entry.get("text", "")
            ocr_description = ocr_entry.get("description", "")
        else:
            ocr_text = ocr_entry
            ocr_description = ""

        base_text = item.get("text", "")
        if ocr_text and ocr_text.strip() and ocr_text != "ERROR_PARSING_JSON":
            combined_text = f"{base_text} {ocr_text}".strip()
        else:
            combined_text = base_text.strip()

        raw_description = ocr_description.strip()

        if raw_description == "ERROR_PARSING_JSON":
            raw_description = ""

        item["raw_text"] = combined_text
        item["clean_text"] = clean_meme_text_for_transformer(combined_text)

        item["raw_description"] = raw_description
        item["clean_description"] = clean_meme_text_for_transformer(raw_description)

        item["text_origin"] = "ocr_merged" if (ocr_text and ocr_text.strip()) else "original"
        item["processed_sensors"] = extract_sensors_for_dl(item)

        processed_list.append(item)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_list, f, indent=4, ensure_ascii=False)

    print(f"Successfully processed {len(processed_list)} items to {output_json_path}.")


def main():
    """
    EXIST2026_training.json
    ocr_results_combined.json
    output to processed_data.json
    """
    parser = argparse.ArgumentParser(
        description="Preprocess meme dataset by merging OCR results and cleaning text for transformer input."
    )
    parser.add_argument("--input-path", type=str, required=True, help="Path to the raw input JSON file")
    parser.add_argument("--ocr-path", type=str, required=True, help="Path to the OCR results JSON file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the processed output JSON file")
    args = parser.parse_args()

    preprocess_data(args.input_path, args.ocr_path, args.output_path)


if __name__ == "__main__":
    main()
