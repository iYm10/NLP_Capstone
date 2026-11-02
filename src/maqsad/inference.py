"""Inference from a locally trained multi-task checkpoint."""
import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .model import MultiTaskModel
from .data import normalize


def predict_transformer(texts, checkpoint="artifacts/marbert-joint"):
    if not texts or any(not str(t).strip() for t in texts):
        raise ValueError("Provide non-empty text")
    path = Path(checkpoint)
    config = json.loads((path / "metrics.json").read_text())["configuration"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = MultiTaskModel(AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True)))
    model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    inputs = tokenizer([normalize(t) for t in texts], padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs)
    ps, pm = logits["sarcasm"].softmax(-1), logits["sentiment"].softmax(-1)
    results = []
    for s, m in zip(ps, pm):
        row = {"sentiment": ["negative", "neutral", "positive"][int(m.argmax())], "sentiment_score": float(m.max())}
        if config["task"] == "joint":
            row.update(sarcasm=bool(s.argmax()), sarcasm_score=float(s[1]))
        results.append(row)
    return results
