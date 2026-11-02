import numpy as np
import pandas as pd
import pytest
from maqsad.data import normalize, load_file
from maqsad.evaluation import evaluate


def test_normalization_preserves_negation_and_emoji():
    assert normalize('  لَا أحبّ هذا 😒  ') == 'لا أحب هذا 😒'
    assert normalize('مرحبا @person https://example.com') == 'مرحبا USER URL'


def test_bad_label_is_rejected(tmp_path):
    file = tmp_path / 'bad.csv'
    pd.DataFrame([dict(tweet='نص', sarcasm='maybe', sentiment='NEG', dialect='msa')]).to_csv(file, index=False)
    with pytest.raises(ValueError, match='sarcasm'):
        load_file(file)


def test_metrics_include_absent_classes():
    frame = pd.DataFrame({'sarcasm_label': [0, 1], 'sentiment_label': [0, 0], 'dialect': ['gulf', 'gulf']})
    result = evaluate(frame, [0, 1], [0, 0])
    assert result['sarcasm_f1'] == 1.0
    assert result['sentiment_macro_f1'] == pytest.approx(1 / 3)
    assert result['by_dialect']['gulf']['n'] == 2


def test_multitask_gradient_and_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    from transformers import BertConfig, BertModel
    from maqsad.model import MultiTaskModel, task_loss
    torch.manual_seed(42)
    model = MultiTaskModel(BertModel(BertConfig(vocab_size=64, hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)))
    tokens = torch.randint(0, 64, (4, 8)); mask = torch.ones_like(tokens)
    out = model(tokens, mask)
    assert out['sarcasm'].shape == (4, 2)
    assert out['sentiment'].shape == (4, 3)
    loss = task_loss(out, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 2, 1]))
    loss.backward()
    assert model.sarcasm.weight.grad.abs().sum() > 0
    assert model.sentiment.weight.grad.abs().sum() > 0
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    task_loss(model(tokens, mask), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 2, 1]), task='sentiment').backward()
    assert model.sarcasm.weight.grad is None
    model.eval()
    before = model(tokens, mask)['sentiment'].detach()
    torch.save(model.state_dict(), tmp_path / 'weights.pt')
    clone = MultiTaskModel(BertModel(model.encoder.config)); clone.eval()
    clone.load_state_dict(torch.load(tmp_path / 'weights.pt', weights_only=True))
    assert torch.allclose(before, clone(tokens, mask)['sentiment'])


def test_training_and_inference_offline(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from argparse import Namespace
    from transformers import BertConfig, BertModel, BertTokenizerFast
    import maqsad.train as training
    from maqsad.inference import predict_transformer
    source = tmp_path / 'tiny-local-bert'; source.mkdir()
    (source / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nخدمة\nجيدة\nسيئة\nعادية\n', encoding='utf-8')
    BertTokenizerFast(vocab_file=str(source / 'vocab.txt')).save_pretrained(source)
    BertModel(BertConfig(vocab_size=9, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24)).save_pretrained(source)
    frame = pd.DataFrame({'text': ['خدمة جيدة', 'خدمة سيئة', 'خدمة عادية'] * 4,
        'sentiment_label': [2, 0, 1] * 4, 'sarcasm_label': [0, 1, 0] * 4, 'dialect': ['msa'] * 12})
    monkeypatch.setattr(training, 'prepare', lambda *args: (frame, frame, frame, {'synthetic_test_only': True}))
    args = Namespace(model=str(source), revision='main', seed=42, epochs=1, batch_size=4,
        max_length=8, learning_rate=1e-3, task='joint', data_dir='unused', output=str(tmp_path / 'checkpoint'))
    report = training.run(args)
    assert report['test']['n'] == 12
    predictions = predict_transformer(['خدمة جيدة'], args.output)
    assert predictions[0]['sentiment'] in ['negative', 'neutral', 'positive']
    assert 0 <= predictions[0]['sarcasm_score'] <= 1
