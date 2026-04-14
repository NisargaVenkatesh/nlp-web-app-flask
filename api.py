import os
import requests


HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "dslim/bert-base-NER"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

def ner(text):
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": text
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

    response.raise_for_status()
    return response.json()
