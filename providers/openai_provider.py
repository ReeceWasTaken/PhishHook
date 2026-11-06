import os, json
from dotenv import load_dotenv
from openai import OpenAI

# Loads API key from .env file
load_dotenv()


# Prompt for API to understand what it needs to do
SYSTEM_PROMPT = (
    "You are a cybersecurity email classification assistant.\n"
    "Classify emails as:\n"
    "1 = phishing (deception, impersonation, malicious links, credential harvesting)\n"
    "0 = legitimate (including newsletters, marketing emails, system notifications).\n\n"
    "Guidelines:\n"
    "- Promotional tone alone is NOT phishing.\n"
    "- Poor grammar alone is NOT phishing.\n"
    "- Classify as phishing if multiple suspicious indicators are present or if there is deceptive intent.\n"
    "Return ONLY JSON: {\"label\":0|1, \"rationale\":\"...\"}"
)

# Classification of emails to 0 or 1 and to give an explanation
def classify_email(text: str) -> tuple[int, str]:
    text = (text or "")[:4000]

# LLM model selection
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. See .env.example")
    client = OpenAI(api_key=key)

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"EMAIL:\n{text}\n\nReturn ONLY JSON."},
        ],
    )

# Extracts raw output for better classifcation
    out = resp.output_text.strip()
    start, end = out.find("{"), out.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Provider response did not contain JSON")

    try:
        obj = json.loads(out[start : end + 1])
        label = 1 if int(obj.get("label", 0)) == 1 else 0
        rationale = str(obj.get("rationale", "")).strip()[:300]
        return label, rationale
    except Exception:
        raise ValueError("Provider response contained invalid JSON")