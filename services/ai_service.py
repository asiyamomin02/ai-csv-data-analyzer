import os
import time
import requests

MODEL_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

def get_api_key():
    return os.getenv("GROQ_API_KEY")

def call_ai(system, user, max_tokens=400):
    api_key = get_api_key()
    
    if not api_key:
        return "AI is disabled because API key is missing."

    if len(user) > 2800:
        user = user[:2800] + "\n...(truncated)"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    for attempt in range(3):
        try:
            r = requests.post(MODEL_URL, headers=headers, json=payload, timeout=30)

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

            if r.status_code == 429:
                time.sleep(4 + attempt * 3)
                continue

            try:
                return f"Error: {r.json().get('error', {}).get('message', 'Unknown')}"
            except Exception:
                return f"Error: HTTP {r.status_code}"

        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return f"Error: {str(e)}"

    return "Rate limit reached. Wait a few seconds and retry."
