import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_translator(
        model_name: str = "facebook/nllb-200-distilled-1.3B"
) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]:
    print(f"Loading NLLB translator: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    return tokenizer, model, device


def translate_batch(texts, tokenizer, model, device, src_lang, tgt_lang):
    """Translates a single batch of texts."""
    if not texts or all(t is None or not str(t).strip() for t in texts):
        return ["" for _ in texts]

    tokenizer.src_lang = src_lang

    try:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    except Exception:
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    clean_texts = [str(t) if t is not None else "" for t in texts]

    encoded = tokenizer(clean_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512
        )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def augment_dataset(input_json_path, output_json_path, batch_size=16):
    print("Loading original dataset...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    is_list = isinstance(data, list)
    print(f"Loaded {len(data)} original memes.")
    checkpoint_path = output_json_path.replace(".json", "_checkpoint.jsonl")
    processed_ids = set()
    augmented_items = []

    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint file: {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_ids.add(item["id_EXIST"])
                    augmented_items.append(item)
        print(f"Resuming progress. Skipped {len(processed_ids)} already translated items.")

    if is_list:
        en_items = [item for item in data if
                    item.get("lang") == "en" and f"{item.get("id_EXIST")}_aug" not in processed_ids]
        es_items = [item for item in data if
                    item.get("lang") == "es" and f"{item.get("id_EXIST")}_aug" not in processed_ids]
    else:
        en_items = [(k, v) for k, v in data.items() if
                    v.get("lang") == "en" and f"{v.get("id_EXIST")}_aug" not in processed_ids]
        es_items = [(k, v) for k, v in data.items() if
                    v.get("lang") == "es" and f"{v.get("id_EXIST")}_aug" not in processed_ids]

    if en_items or es_items:
        tokenizer, model, device = load_translator()

        with open(checkpoint_path, "a", encoding="utf-8") as ckpt_f:

            def process_and_write(items, src_lang, tgt_lang, target_lang_label):
                if not items: return

                for i in tqdm(range(0, len(items), batch_size), desc=f"Translating {src_lang} -> {tgt_lang}"):
                    batch = items[i:i + batch_size]

                    if is_list:
                        batch_items = batch
                    else:
                        batch_items = [v for k, v in batch]

                    clean_texts = [item.get("clean_text", "") for item in batch_items]
                    clean_descriptions = [item.get("clean_description", "") for item in batch_items]

                    trans_clean_texts = translate_batch(clean_texts, tokenizer, model, device, src_lang, tgt_lang)
                    trans_clean_descriptions = translate_batch(clean_descriptions, tokenizer, model, device, src_lang,
                                                               tgt_lang)

                    for j in range(len(batch_items)):
                        orig_item = batch_items[j]
                        new_item = orig_item.copy()

                        orig_id = orig_item.get("id_EXIST", f"unknown_{i + j}")
                        new_item["id_EXIST"] = f"{orig_id}_aug"
                        new_item["lang"] = target_lang_label

                        if "clean_text" in orig_item:
                            new_item["clean_text"] = trans_clean_texts[j]
                        if "clean_description" in orig_item:
                            new_item["clean_description"] = trans_clean_descriptions[j]

                        augmented_items.append(new_item)

                        ckpt_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

                    ckpt_f.flush()

            if en_items:
                print(f"\n--- Processing {len(en_items)} English to Spanish items ---")
                process_and_write(en_items, "eng_Latn", "spa_Latn", "es")

            if es_items:
                print(f"\n--- Processing {len(es_items)} Spanish to English items ---")
                process_and_write(es_items, "spa_Latn", "eng_Latn", "en")
    else:
        print("\nAll items have already been translated! Skipping straight to compiling the final file.")

    print("\nCompiling final JSON file...")

    if is_list:
        final_data = data + augmented_items
    else:
        final_data = data.copy()
        for item in augmented_items:
            final_data[item["id_EXIST"]] = item

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"Data multiplication complete! Saved {len(final_data)} total memes to: {output_json_path}")


if __name__ == "__main__":
    from exist_2026.path_manager import PathManager

    INPUT_FILE = PathManager.DATA_EXIST_DIR / "training" / "processed_data.json"
    OUTPUT_FILE = PathManager.DATA_EXIST_DIR / "training" / "processed_data_augmented.json"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    augment_dataset(INPUT_FILE, OUTPUT_FILE)
