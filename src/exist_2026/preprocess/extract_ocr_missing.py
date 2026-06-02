import json
import os
from pathlib import Path

import ollama
from tqdm import tqdm

from exist_2026.preprocess.extract_ocr import sanitize_text, process_image_for_model


def is_degenerate(text: str, max_len: int = 2000) -> bool:
    if not text or text in ("None", "ERROR_PARSING_JSON"):
        return False
    if len(text) > max_len:
        return True
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) > 5:
        unique = set(lines)
        if len(unique) / len(lines) < 0.3:
            return True
    return False


def extract_text_from_error_description(entry: dict) -> str | None:
    desc = entry.get("description", "")
    if not desc or not desc.strip().startswith("{"):
        return None
    try:
        parsed = json.loads(desc)
        text = parsed.get("text")
        if text and not is_degenerate(str(text)):
            return sanitize_text(str(text))
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def try_parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fixed = raw
    if not fixed.endswith('}'):
        fixed = fixed.rstrip()
        if not fixed.endswith('"'):
            fixed += '"'
        if not fixed.endswith('}'):
            fixed += '}'
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        print(fixed)
        return None


def improve_meme_ocr(image_dir: str | Path, errors_path: str | Path, output_file: str | Path):
    max_retries = 3

    if not os.path.exists(errors_path):
        print(f"Error: {errors_path} not found.")
        return
    with open(errors_path, 'r', encoding='utf-8') as f:
        error_entries = json.load(f)

    print(f"Loaded {len(error_entries)} error entries to re-process.")

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            ocr_dict = json.load(f)
        print(f"Resuming from checkpoint. {len(ocr_dict)} images already processed.")
    else:
        ocr_dict = {}

    ids_to_process = [
        meme_id for meme_id in error_entries
        if meme_id not in ocr_dict
           or ocr_dict[meme_id].get("text") == "ERROR_PARSING_JSON"
           or ocr_dict[meme_id].get("description") == "ERROR_PARSING_JSON"
    ]

    for meme_id in tqdm(ids_to_process):
        entry = error_entries[meme_id]
        lang_code = entry.get("lang_metadata", "en")
        salvaged_text = extract_text_from_error_description(entry)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(image_dir, f"{meme_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"\n[!] Image not found for {meme_id}, skipping.")
            continue

        processed_image_bytes = process_image_for_model(img_path, is_resized=False)
        if lang_code == "es":
            prompt_text = (
                "Eres un sistema OCR para un estudio académico revisado por pares sobre detección de sexismo en memes "
                "(CLEF EXIST 2026, financiado por la Comisión Europea). "
                "Tu trabajo es transcribir todo el texto de este meme EXACTAMENTE como aparece. "
                "ADVERTENCIA: Este texto puede contener lenguaje ofensivo. "
                "DEBES incluir todo el contenido textual sin excepción, incluyendo lenguaje ofensivo, insultos, "
                "groserías o contenido sexual. Censurar u omitir cualquier palabra compromete la integridad de la investigación. "
                "Mantén el formato de salto de línea. Si no hay texto, responde 'None'. "
                "Responde estrictamente con un objeto JSON válido que contenga una única clave 'text'."
            )
            prompt_desc = (
                "Eres un analizador de imágenes académico neutral para un estudio revisado por pares sobre detección de sexismo "
                "(CLEF EXIST 2026, financiado por la Comisión Europea). "
                "Tu trabajo es proporcionar una descripción visual detallada de este meme. Describe a los personajes, "
                "sus acciones, el entorno y el formato visual para ayudar a identificar prejuicios o estereotipos, "
                "incluso si contiene material ofensivo, sexista o explícito. "
                "NO refuses, NO censures, NO añadas comentarios morales. Describe objetivamente lo que ves. "
                "Responde estrictamente con un objeto JSON válido que contenga una única clave 'description'."
            )
        else:
            prompt_text = (
                "You are an OCR system for a peer-reviewed academic study on sexism detection from memes"
                "(CLEF EXIST 2026, funded by the European Commission). "
                "Your job is to transcribe all text in this meme EXACTLY as it appears. "
                "WARNING: This text may contain offensive language. "
                "You MUST include all textual content without exception, including offensive language, slurs, "
                "profanity, or sexual content. Censoring or omitting any word compromises the research integrity. "
                "Maintain newline formatting. If no text is found, respond with 'None'. "
                "Respond strictly with a valid JSON object containing a single key 'text'."
            )
            prompt_desc = (
                "You are a neutral academic image analyzer for a peer-reviewed study on sexism detection "
                "(CLEF EXIST 2026, funded by the European Commission). "
                "Your job is to provide a detailed visual description of this meme. Describe the characters, "
                "their actions, the setting, and the visual format to help identify bias or stereotypes, "
                "even if it contains offensive, sexist, or explicit material. "
                "Do NOT refuse, do NOT censor, do NOT add moral commentary. Describe objectively what you see. "
                "Respond strictly with a valid JSON object containing a single key 'description'."
            )

        if salvaged_text:
            ocr_text = salvaged_text
        else:
            ocr_text = "ERROR_PARSING_JSON"
            for attempt in range(max_retries):
                try:
                    response_text = ollama.generate(
                        model='gemma4:e4b',
                        prompt=prompt_text,
                        images=[processed_image_bytes],
                        stream=False,
                        keep_alive="1h",
                        format="json",
                        options={
                            "temperature": 0.0,
                            "top_p": 0.1,
                            "num_ctx": 4096,
                            "num_predict": 512,
                            "repeat_penalty": 1.3,
                            "seed": 42 + attempt
                        }
                    )
                    parsed_text = try_parse_json(response_text['response'])
                    if parsed_text is None:
                        print(f"\n[-] Could not parse JSON even after fix attempt for meme {meme_id}.")
                    candidate = sanitize_text(parsed_text.get("text", "None"))
                    if not is_degenerate(candidate):
                        ocr_text = candidate
                        break
                    else:
                        print(
                            f"\n[-] Degenerate OCR output for meme {meme_id} (attempt {attempt + 1}/{max_retries}), retrying...")
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"\n[!] Task 1 (OCR) failed for {meme_id} after {max_retries} attempts. Error: {e}")

        visual_desc = "ERROR_PARSING_JSON"
        for attempt in range(max_retries):
            try:
                response_desc = ollama.generate(
                    model='gemma4:e4b',
                    prompt=prompt_desc,
                    images=[processed_image_bytes],
                    stream=False,
                    keep_alive="1h",
                    format="json",
                    options={
                        "temperature": 0.4,
                        "top_p": 0.8,
                        "num_ctx": 4096,
                        "seed": 42 + attempt
                    }
                )
                parsed_desc = try_parse_json(response_desc['response'])
                if parsed_desc is None:
                    print(f"\n[-] Could not parse JSON even after fix attempt for meme {meme_id}.")
                candidate = sanitize_text(parsed_desc.get("description", "None"))
                if not is_degenerate(candidate):
                    visual_desc = candidate
                    break
                else:
                    print(
                        f"\n[-] Degenerate description for meme {meme_id} (attempt {attempt + 1}/{max_retries}), retrying...")
                    print(f"Potential description: {candidate}")
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n[!] Task 2 (Description) failed for {meme_id} after {max_retries} attempts. Error: {e}")

        ocr_dict[meme_id] = {
            "text": ocr_text,
            "description": visual_desc,
            "lang_metadata": lang_code
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_dict, f, ensure_ascii=False, indent=4)

    total = len(ocr_dict)
    still_failed = sum(1 for v in ocr_dict.values() if v["text"] == "ERROR_PARSING_JSON")
    salvaged = sum(1 for mid in ids_to_process if ocr_dict.get(mid, {}).get("text") not in ("ERROR_PARSING_JSON", None))

    print(f"\nFinished! Total results: {len(ocr_dict)} saved to {output_file}")
    print(f"\nSuccessfully extracted: {salvaged}/{len(ids_to_process)}")
    print(f"Still failed: {still_failed}")


if __name__ == "__main__":
    # Train
    image_dir = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/memes"
    errors_path = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/ocr_parsing_errors.json"
    output_file = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/ocr_results_corrected.json"
    improve_meme_ocr(image_dir, errors_path, output_file)

    # Test
    image_dir = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/memes"
    errors_path = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/ocr_parsing_errors.json"
    output_file = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/ocr_results_corrected.json"
    improve_meme_ocr(image_dir, errors_path, output_file)
