"""
Step 3: Translate English transcript to Hindi.

Primary: IndicTrans2 (ai4bharat) — best quality for Indian languages.
Fallback: googletrans — lightweight, no VRAM needed.

Usage:
    python scripts/03_translate.py
"""

import json
import os
import argparse
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def translate_indictrans2(text: str, src_lang: str, tgt_lang: str, model_name: str) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransTokenizer import IndicProcessor

    print(f"Loading IndicTrans2 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")

    with __import__("torch").no_grad():
        generated = model.generate(**inputs, num_beams=5, max_length=256)

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    result = ip.postprocess_batch(decoded, lang=tgt_lang)
    return result[0]


def translate_googletrans(text: str, tgt_lang: str) -> str:
    from googletrans import Translator
    translator = Translator()
    result = translator.translate(text, dest=tgt_lang)
    return result.text


def main():
    parser = argparse.ArgumentParser(description="Step 3: Translate English to Hindi")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    transcript_path = os.path.join(cfg["output"]["transcripts_dir"], "transcript.json")
    output_dir = cfg["output"]["translations_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}. Run 02_transcribe.py first.")

    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    english_text = transcript["text"].strip()
    print(f"English text:\n  {english_text}\n")

    use_indictrans2 = cfg["translation"]["use_indictrans2"]
    hindi_text = None

    if use_indictrans2:
        try:
            hindi_text = translate_indictrans2(
                text=english_text,
                src_lang=cfg["translation"]["source_lang"],
                tgt_lang=cfg["translation"]["target_lang"],
                model_name=cfg["translation"]["indictrans2_model"]
            )
            method = "IndicTrans2"
        except Exception as e:
            print(f"IndicTrans2 failed: {e}\nFalling back to googletrans...")

    if hindi_text is None:
        hindi_text = translate_googletrans(english_text, cfg["translation"]["target_lang_short"])
        method = "googletrans"

    print(f"Hindi text ({method}):\n  {hindi_text}\n")

    # Save translated text
    output_path = os.path.join(output_dir, "translation.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(hindi_text)

    # Save metadata (method used, source, target)
    meta_path = os.path.join(output_dir, "translation_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": method,
            "source_lang": cfg["translation"]["source_lang"],
            "target_lang": cfg["translation"]["target_lang"],
            "english_text": english_text,
            "hindi_text": hindi_text
        }, f, ensure_ascii=False, indent=2)

    print("Step 3 complete.")
    print(f"  Translation : {output_path}")
    print(f"  Metadata    : {meta_path}")


if __name__ == "__main__":
    main()
