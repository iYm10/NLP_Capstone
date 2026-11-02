import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score


def evaluate(frame, sarcasm, sentiment):
    sarcasm, sentiment = np.asarray(sarcasm), np.asarray(sentiment)
    result = {
        "n": len(frame),
        "sarcasm_f1": float(f1_score(frame.sarcasm_label, sarcasm, zero_division=0)),
        "sarcasm_macro_f1": float(f1_score(frame.sarcasm_label, sarcasm, labels=[0, 1], average="macro", zero_division=0)),
        "sentiment_macro_f1": float(f1_score(frame.sentiment_label, sentiment, labels=[0, 1, 2], average="macro", zero_division=0)),
        "sentiment_accuracy": float(accuracy_score(frame.sentiment_label, sentiment)),
        "sarcasm_report": classification_report(frame.sarcasm_label, sarcasm, labels=[0, 1], target_names=["not_sarcastic", "sarcastic"], output_dict=True, zero_division=0),
        "sentiment_report": classification_report(frame.sentiment_label, sentiment, labels=[0, 1, 2], target_names=["negative", "neutral", "positive"], output_dict=True, zero_division=0),
    }
    result["by_dialect"] = {}
    for dialect in sorted(frame.dialect.unique()):
        mask = frame.dialect.eq(dialect).to_numpy()
        result["by_dialect"][dialect] = {
            "n": int(mask.sum()),
            "sarcasm_f1": float(f1_score(frame.loc[mask, "sarcasm_label"], sarcasm[mask], zero_division=0)),
            "sentiment_macro_f1": float(f1_score(frame.loc[mask, "sentiment_label"], sentiment[mask], labels=[0, 1, 2], average="macro", zero_division=0)),
        }
    return result
