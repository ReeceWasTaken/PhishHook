import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from providers.openai_provider import classify_email as openai_classify
from providers.gemini_provider import classify_email as gemini_classify

# This script file is what runs they hyrbid model Baseline + OpenAI + Gemini

load_dotenv()

DATASET_PATH = "phishing_email.csv"
TEXT_COL = "text_combined"
LABEL_COL = "label"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TEST_SAMPLES = 150
SLEEP_SECONDS = 0.2

OPENAI_CACHE = "caches/hybrid_openai_cache.jsonl"
GEMINI_CACHE = "caches/hybrid_gemini_cache.jsonl"

# Reads cached predictions and loads it into memory

def load_cache(path: str) -> dict:
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cache[str(rec["id"])] = rec
            except json.JSONDecodeError:
                continue
    return cache

# Adds a new prediction to the cache file
def append_cache(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def majority_vote(preds: list[int]) -> int:
    """
    Majority vote:
    If 2 or more models predict phishing (1), return 1.
    Otherwise return 0.
    """
    return 1 if sum(preds) >= 2 else 0


def main():
    # Loads dataset
    df = pd.read_csv(DATASET_PATH)
    X = df[TEXT_COL]
    y = df[LABEL_COL]

    # Same split for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Limits sample size for API
    X_test = X_test.iloc[:MAX_TEST_SAMPLES]
    y_test = y_test.iloc[:MAX_TEST_SAMPLES]

    # Baseline Logistic Regression
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_vec, y_train)

    baseline_preds = baseline_model.predict(X_test_vec)

    # Load caches for API models
    openai_cache = load_cache(OPENAI_CACHE)
    gemini_cache = load_cache(GEMINI_CACHE)

    openai_preds = []
    gemini_preds = []
    hybrid_preds = []

    openai_rationales = []
    gemini_rationales = []

    # Run hybrid evaluation
    for idx, text, base_pred in zip(X_test.index, X_test.values, baseline_preds):
        key = str(idx)

        # OpenAI prediction
        if key in openai_cache:
            openai_pred = int(openai_cache[key]["pred"])
            openai_rationale = openai_cache[key].get("rationale", "")
        else:
            openai_pred, openai_rationale = openai_classify(text)
            append_cache(
                OPENAI_CACHE,
                {"id": key, "pred": int(openai_pred), "rationale": openai_rationale},
            )
            time.sleep(SLEEP_SECONDS)

        # Gemini prediction
        if key in gemini_cache:
            gemini_pred = int(gemini_cache[key]["pred"])
            gemini_rationale = gemini_cache[key].get("rationale", "")
        else:
            gemini_pred, gemini_rationale = gemini_classify(text)
            append_cache(
                GEMINI_CACHE,
                {"id": key, "pred": int(gemini_pred), "rationale": gemini_rationale},
            )
            time.sleep(SLEEP_SECONDS)

        # Save individual predictions
        openai_preds.append(openai_pred)
        gemini_preds.append(gemini_pred)
        openai_rationales.append(openai_rationale)
        gemini_rationales.append(gemini_rationale)

        # Majority vote rule
        final_pred = majority_vote([int(base_pred), int(openai_pred), int(gemini_pred)])
        hybrid_preds.append(final_pred)

    # Print results
    print("Hybrid Model Results:")
    print("Accuracy:", accuracy_score(y_test, hybrid_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, hybrid_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, hybrid_preds))

    # Save output
    os.makedirs("results", exist_ok=True)

    out = pd.DataFrame(
        {
            "id": X_test.index.astype(str),
            "true": y_test.values,
            "baseline_pred": baseline_preds,
            "openai_pred": openai_preds,
            "gemini_pred": gemini_preds,
            "final_vote": hybrid_preds,
            "openai_rationale": openai_rationales,
            "gemini_rationale": gemini_rationales,
        }
    )

    out["ensemble_corrected_error"] = (
        (out["baseline_pred"] != out["true"]) & (out["final_vote"] == out["true"])
    )

    out.to_csv("results/hybrid_predictions.csv", index=False)

    # Error file output
    out["error_type"] = None
    out.loc[(out["true"] == 0) & (out["final_vote"] == 1), "error_type"] = "False Positive"
    out.loc[(out["true"] == 1) & (out["final_vote"] == 0), "error_type"] = "False Negative"

    errors = out[out["error_type"].notnull()]
    errors.to_csv("results/hybrid_errors.csv", index=False)

    print("\nSaved: results/hybrid_predictions.csv")
    print("Saved: results/hybrid_errors.csv")


if __name__ == "__main__":
    main()