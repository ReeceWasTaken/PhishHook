import os, json, time, argparse
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# This script file runs the individual models (OpenAI and Gemini)
load_dotenv()

DATASET_PATH = "phishing_email.csv"
TEXT_COL = "text_combined"
LABEL_COL = "label"

TEST_SIZE = 0.2
RANDOM_STATE = 42
SLEEP_SECONDS = 0.2

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
            except Exception:
                continue
    return cache

# Adds a new prediction to the cache file
def append_cache(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# When running file must enter in provider example: --provider gemini --samples 150
def get_provider(provider_name: str):
    provider_name = provider_name.lower().strip()

    if provider_name == "openai":
        from providers.openai_provider import classify_email
        return "openai", classify_email

    if provider_name == "gemini":
        from providers.gemini_provider import classify_email
        return "gemini", classify_email

    raise ValueError("provider must be one of: openai, gemini")

# Runs script as agrument to avoid any error example: python eval_provider.py --provider openai --samples 150
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openai", "gemini"])
    ap.add_argument("--samples", type=int, default=150)
    args = ap.parse_args()

# Picks correct model and function from whatever the user input
    provider_name, classify_fn = get_provider(args.provider)

# Loads dataset
    df = pd.read_csv(DATASET_PATH)
    X = df[TEXT_COL]
    y = df[LABEL_COL]

    # Splits data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_test = X_test.iloc[: args.samples]
    y_test = y_test.iloc[: args.samples]

# Loads previous predictions
    cache_path = os.path.join("caches", f"{provider_name}_cache.jsonl")
    cache = load_cache(cache_path)

    preds = []
    rationales = []

# Loop through each email so that every email is analyazed and returns a prediction
    for idx, text in zip(X_test.index, X_test.values):
        key = str(idx)

        if key in cache:
            preds.append(int(cache[key]["pred"]))
            rationales.append(cache[key].get("rationale", ""))
            continue

        pred, rationale = classify_fn(text)

        preds.append(int(pred))
        rationales.append(str(rationale))

        append_cache(cache_path, {
            "id": key,
            "pred": int(pred),
            "rationale": str(rationale)
        })

        time.sleep(SLEEP_SECONDS)

    # Prints metrics onto terminal
    print(f"\n=== Provider: {provider_name} | Samples: {len(y_test)} ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save outputs
    out = pd.DataFrame(
        {
            "id": X_test.index.astype(str),
            "true": y_test.values,
            "pred": preds,
            "rationale": rationales,
        }
    )

    os.makedirs("results", exist_ok=True)
    pred_path = os.path.join("results", f"{provider_name}_predictions.csv")
    out.to_csv(pred_path, index=False)

    # Saves Errors to errors.csv file
    out["error_type"] = None
    out.loc[(out["true"] == 0) & (out["pred"] == 1), "error_type"] = "False Positive"
    out.loc[(out["true"] == 1) & (out["pred"] == 0), "error_type"] = "False Negative"

    errs = out[out["error_type"].notnull()]
    err_path = os.path.join("results", f"{provider_name}_errors.csv")
    errs.to_csv(err_path, index=False)

    print(f"\nSaved: {pred_path}")
    print(f"Saved: {err_path}")
    print(f"Cache: {cache_path}")


if __name__ == "__main__":
    main()