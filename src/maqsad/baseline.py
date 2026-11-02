"""CPU baseline: shared character TF-IDF, independent linear classifiers."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import prepare, normalize
from .evaluation import evaluate


def train_baseline(data_dir="data/raw", output="artifacts/baseline.joblib", report_dir="reports"):
    train, val, test, audit = prepare(data_dir)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True, dtype=np.float32)
    x = vectorizer.fit_transform(train.text)
    xv, xt = vectorizer.transform(val.text), vectorizer.transform(test.text)
    models, selection = {}, {}
    for task in ["sarcasm", "sentiment"]:
        best_score, best_model = -1, None
        from sklearn.metrics import f1_score
        for c in [0.5, 2.0, 8.0]:
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=500, solver="liblinear", random_state=42)
            # One-vs-rest explicitly supports multiclass across sklearn versions.
            if task == "sentiment":
                from sklearn.multiclass import OneVsRestClassifier
                model = OneVsRestClassifier(model)
            model.fit(x, train[f"{task}_label"])
            score = f1_score(val[f"{task}_label"], model.predict(xv), average="binary" if task == "sarcasm" else "macro", zero_division=0)
            selection[f"{task}_C={c}"] = float(score)
            if score > best_score:
                best_score, best_model = score, model
        models[task] = best_model
    bundle = {"vectorizer": vectorizer, **models, "type": "tfidf_independent_heads"}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    results = {"model": bundle["type"], "audit": audit, "validation_selection": selection,
        "validation": evaluate(val, models["sarcasm"].predict(xv), models["sentiment"].predict(xv)),
        "test": evaluate(test, models["sarcasm"].predict(xt), models["sentiment"].predict(xt))}
    majority_s = np.full(len(test), train.sarcasm_label.mode().iloc[0])
    majority_m = np.full(len(test), train.sentiment_label.mode().iloc[0])
    results["majority_test"] = evaluate(test, majority_s, majority_m)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir, "baseline_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Do not publish raw user-authored tweets or tweet-level predictions.
    return results


def predict(texts, path="artifacts/baseline.joblib"):
    if not texts or any(not str(t).strip() for t in texts):
        raise ValueError("Provide non-empty Arabic text")
    model = joblib.load(path)  # Load only trusted locally produced artifacts.
    x = model["vectorizer"].transform([normalize(t) for t in texts])
    ps, pm = model["sarcasm"].predict_proba(x), model["sentiment"].predict_proba(x)
    return [{"sarcasm": bool(model["sarcasm"].classes_[a.argmax()]),
             "sarcasm_score": float(a[list(model["sarcasm"].classes_).index(1)]),
             "sentiment": ["negative", "neutral", "positive"][int(model["sentiment"].classes_[b.argmax()])],
             "sentiment_score": float(b.max()), "model": model["type"]} for a, b in zip(ps, pm)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    print(json.dumps(train_baseline(args.data_dir)["test"], indent=2))
