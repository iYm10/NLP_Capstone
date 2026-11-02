"""Fine-tune MARBERT; select checkpoints on validation sentiment macro-F1."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from .data import prepare
from .evaluation import evaluate
from .model import MultiTaskModel, task_loss

class EncodedTweets(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokens = tokenizer(self.frame.text.tolist(), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.tokens.items()},
                "sarcasm_label": torch.tensor(int(self.frame.iloc[i].sarcasm_label)),
                "sentiment_label": torch.tensor(int(self.frame.iloc[i].sentiment_label))}


def predict_loader(model, loader, device):
    model.eval()
    sarcasm, sentiment = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if not k.endswith("_label")}
            outputs = model(**inputs)
            sarcasm.extend(outputs["sarcasm"].argmax(-1).cpu().tolist())
            sentiment.extend(outputs["sentiment"].argmax(-1).cpu().tolist())
    return sarcasm, sentiment


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2:
        raise ValueError("epochs, batch size and sequence length must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train, val, test, audit = prepare(args.data_dir, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = MultiTaskModel.from_pretrained(args.model, args.revision).to(device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(EncodedTweets(train, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EncodedTweets(val, tokenizer, args.max_length), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history, best = [], -1
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for batch in train_loader:
            ys = batch.pop("sarcasm_label").to(device)
            ym = batch.pop("sentiment_label").to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            loss = task_loss(outputs, ys, ym, task=args.task)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += loss.item() * len(ys)
        ps, pm = predict_loader(model, val_loader, device)
        metrics = evaluate(val, ps, pm)
        score = metrics["sentiment_macro_f1"]
        history.append({"epoch": epoch + 1, "train_loss": total / len(train), "validation_sentiment_macro_f1": score})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), output / "weights.pt")
    model.load_state_dict(torch.load(output / "weights.pt", map_location=device, weights_only=True))
    tokenizer.save_pretrained(output)
    model.encoder.config.save_pretrained(output)
    test_loader = DataLoader(EncodedTweets(test, tokenizer, args.max_length), batch_size=args.batch_size)
    ps, pm = predict_loader(model, test_loader, device)
    result = evaluate(test, ps, pm)
    if args.task == "sentiment":
        # Sarcasm head is untrained in the ablation; do not publish its scores.
        result = {k: v for k, v in result.items() if not k.startswith("sarcasm")}
        for group in result["by_dialect"].values():
            group.pop("sarcasm_f1", None)
    report = {"configuration": vars(args), "resolved_model_commit": getattr(model.encoder.config, "_commit_hash", None), "device": str(device), "audit": audit, "history": history, "test": result}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="UBC-NLP/MARBERT")
    p.add_argument("--revision", default="main", help="Pin a model commit for strict repeatability")
    p.add_argument("--task", choices=["joint", "sentiment"], default="joint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output", default="artifacts/marbert-joint")
    print(json.dumps(run(p.parse_args())["test"], indent=2))

if __name__ == "__main__":
    main()
