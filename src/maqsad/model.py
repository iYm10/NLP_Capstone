"""Shared transformer encoder with sarcasm and sentiment classification heads."""
import torch
from torch import nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.sarcasm = nn.Linear(encoder.config.hidden_size, 2)
        self.sentiment = nn.Linear(encoder.config.hidden_size, 3)

    @classmethod
    def from_pretrained(cls, name, revision="main"):
        return cls(AutoModel.from_pretrained(name, revision=revision))

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state[:, 0]
        hidden = self.dropout(hidden)
        return {"sarcasm": self.sarcasm(hidden), "sentiment": self.sentiment(hidden)}


def task_loss(outputs, sarcasm, sentiment, task="joint", weight=1.0):
    loss = nn.functional.cross_entropy(outputs["sentiment"], sentiment)
    if task == "joint":
        loss = loss + weight * nn.functional.cross_entropy(outputs["sarcasm"], sarcasm)
    elif task != "sentiment":
        raise ValueError("task must be joint or sentiment")
    return loss
