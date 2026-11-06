# PhishHook

**Hybrid phishing-email detection using classical machine learning and two LLM classifiers.**

PhishHook evaluates email text using a TF-IDF + logistic regression baseline, OpenAI, and Google Gemini. The hybrid evaluation uses a simple **2-of-3 majority vote** to decide whether a message is phishing (`1`) or legitimate (`0`).

> Portfolio research prototype, not a production email security.

## Architecture

```text
Email dataset (text_combined, label)
              |
        Train/test split
              |
     +--------+--------+
     |        |        |
 TF-IDF +   OpenAI   Gemini
 Logistic      |        |
 Regression    |        |
     +--------+--------+
              |
       Majority vote
              |
       Metrics / CSV output
```

## Setup

Requires Python 3.11+ and an email classification CSV you are authorized to use.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and replace placeholders with your own credentials. **Never commit `.env` or share real API keys.** API calls may incur charges.

Place `phishing_email.csv` in the repository root. Required columns:

| Column | Meaning |
|---|---|
| `text_combined` | Email text to classify |
| `label` | `0` legitimate; `1` phishing |

The original dataset is intentionally **not included** in this public-ready package because its redistribution rights and potential personal information have not been verified. Document the dataset's source and license before distributing it.

## Run

```bash
# Classical ML baseline (no API key needed)
python main.py

# Evaluate one LLM on up to 150 test emails
python eval_provider.py --provider openai --samples 150
python eval_provider.py --provider gemini --samples 150

# Evaluate all three classifiers with majority voting
python hybrid_eval.py
```

`hybrid_eval.py` currently evaluates the first 150 test records (see `MAX_TEST_SAMPLES`). The scripts use an 80/20 stratified split with random state 42. Caches and generated CSV results are stored locally under `caches/` and `results/` and ignored by Git.

## Security and limitations

- Keys are loaded from environment variables, not hard-coded. `.env` is Git-ignored.
- **Do not send real private emails to third-party model APIs** without consent and an appropriate data-handling review.
- LLM output parsing is basic; malformed responses now raise an error rather than silently marking a message legitimate.
- Predictions may vary with model versions, prompts, and API responses.
- The baseline is trained on the training partition only; evaluation uses the held-out partition.
- The hybrid evaluation is not a production deployment, real-time scanner, or independently verified accuracy guarantee.
- The project does not yet include automated tests, robust retries, or reproducible frozen dependencies.

## Repository contents

```text
phishhook/
├── providers/
│   ├── __init__.py
│   ├── openai_provider.py
│   └── gemini_provider.py
├── .env.example
├── .gitignore
├── main.py
├── eval_provider.py
├── hybrid_eval.py
├── requirements.txt
└── README.md
```

## Future improvements

Add a command-line prediction interface, automated tests, provider retries and structured-output validation, reproducible benchmarks, and a web dashboard.
