import io
import json
import os
from pathlib import Path

import ollama
from PIL import Image
from tqdm import tqdm


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')


def process_image_for_model(img_path: str | Path, max_dim: int = 1024, is_resized: bool = False) -> bytes:
    """
    Resizes the image to a max dimension and pads it to a square
    to prevent aspect ratio distortion and context window bloat.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")

        if is_resized:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        current_w, current_h = img.size
        new_size = max(current_w, current_h)

        square_img = Image.new("RGB", (new_size, new_size), (255, 255, 255))

        offset_x = (new_size - current_w) // 2
        offset_y = (new_size - current_h) // 2
        square_img.paste(img, (offset_x, offset_y))

        img_byte_arr = io.BytesIO()
        square_img.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()


def improve_meme_ocr(image_dir: str | Path, metadata_path: str | Path, output_file: str | Path):
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Metadata file {metadata_path} not found. Defaulting to English.")
        metadata = {}

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            ocr_dict = json.load(f)
        print(f"Resuming from checkpoint. {len(ocr_dict)} images already processed.")
    else:
        ocr_dict = {}

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images_to_process = [f for f in image_files if f.split('.')[0] not in ocr_dict]

    print(f"Processing {len(images_to_process)} remaining memes...")

    for _i, img_name in enumerate(tqdm(images_to_process)):
        meme_id = img_name.split('.')[0]
        img_path = os.path.join(image_dir, img_name)

        lang_code = metadata.get(meme_id, {}).get("lang", "en")

        processed_image_bytes = process_image_for_model(img_path, max_dim=1024, is_resized=True)

        if lang_code == "es":
            prompt = (
                "Actúa como un sistema OCR experto y analizador de imágenes para una investigación académica sobre la detección de sexismo. "
                "Analiza este meme y proporciona la siguiente información en formato JSON:\n"
                "1. 'text': Transcribe todo el texto del meme EXACTAMENTE como aparece. "
                "ADVERTENCIA: Este texto puede contener lenguaje ofensivo, groserías o insultos. "
                "NO censures, omitas ni filtres ninguna palabra. Mantén el formato de salto de línea. Ignora marcas de agua. Si no hay texto, responde 'None'.\n"
                "2. 'description': Proporciona una descripción visual detallada. Describe a los personajes, sus acciones, el entorno y el formato del meme "
                "(por ejemplo, pantalla dividida, texto de impacto) para ayudar a identificar prejuicios, estereotipos o sexismo implícito.\n"
                "Responde estrictamente con un objeto JSON válido que contenga las claves 'text' y 'description'."
            )
        else:
            prompt = (
                "Act as an expert OCR system and image analyzer for an academic research project on sexism detection. "
                "Analyze this meme and provide the following information in JSON format:\n"
                "1. 'text': Transcribe all text in the meme EXACTLY as it appears. "
                "WARNING: This text may contain highly offensive language, profanity, or slurs. "
                "DO NOT censor, omit, or filter any words. Maintain newline formatting. Ignore watermarks. If no text is found, respond with 'None'.\n"
                "2. 'description': Provide a detailed visual description. Describe the characters, their actions, the setting, and the meme format "
                "(e.g., split screen, impact font) to help identify bias, stereotypes, or implicit sexism.\n"
                "Respond strictly with a valid JSON object containing the keys 'text' and 'description'."
            )

        try:
            response = ollama.generate(
                model='gemma4:e4b',
                prompt=prompt,
                images=[processed_image_bytes],
                stream=False,
                keep_alive="1h",
                format="json",
                options={
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "num_ctx": 2048,
                    "seed": 42
                }
            )

            raw_response = response['response'].strip()

            try:
                parsed_result = json.loads(raw_response)
                ocr_text = sanitize_text(parsed_result.get("text", "None"))
                visual_desc = sanitize_text(parsed_result.get("description", "None"))
            except json.JSONDecodeError:
                ocr_text = "ERROR_PARSING_JSON"
                visual_desc = sanitize_text(raw_response)

            ocr_dict[meme_id] = {
                "text": ocr_text,
                "description": visual_desc,
                "lang_metadata": lang_code
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_dict, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"\nError processing {meme_id}: {e}")

    print(f"\nSuccessfully finished! Total results: {len(ocr_dict)} saved to {output_file}")


if __name__ == "__main__":
    # Test:
    image_dir = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/memes"
    metadata_path = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/EXIST2026_test_clean.json"
    output_file = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/test/ocr_results.json"
    improve_meme_ocr(image_dir, metadata_path, output_file)

    # Train
    image_dir = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/memes"
    metadata_path = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/EXIST2026_training.json"
    output_file = "/data/Medz/EXIST 2026 Dataset V0.2/EXIST 2026 Memes Dataset/training/ocr_results.json"
    improve_meme_ocr(image_dir, metadata_path, output_file)
