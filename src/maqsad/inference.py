"""Inference from a locally trained multi-task checkpoint."""
import json
from functools import lru_cache
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .model import MultiTaskModel
from .data import normalize


@lru_cache(maxsize=1)
def load_checkpoint(checkpoint):
    """Reuse one read-only model across requests; never cache user inputs."""
    path = Path(checkpoint)
    config = json.loads((path / "metrics.json").read_text())["configuration"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = MultiTaskModel(AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True)))
    model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True, mmap=True), assign=True)
    model.eval()
    return config, tokenizer, model


def predict_transformer(texts, checkpoint="artifacts/marbert-joint"):
    if not texts or any(not str(t).strip() for t in texts):
        raise ValueError("Provide non-empty text")
    config, tokenizer, model = load_checkpoint(str(Path(checkpoint).resolve()))
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
