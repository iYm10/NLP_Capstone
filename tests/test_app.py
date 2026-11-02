from pathlib import Path
from streamlit.testing.v1 import AppTest
import maqsad.baseline


def open_app(monkeypatch):
    def predict(texts, path):
        return [dict(sarcasm=False,sarcasm_score=0.15,sentiment='positive',sentiment_score=0.8,model='test') for _ in texts]
    monkeypatch.setattr(maqsad.baseline,'predict',predict)
    # Only the checkpoint-presence check is stubbed; UI interactions run normally.
    original=Path.exists
    monkeypatch.setattr(Path,'exists',lambda p: True if p.name=='baseline.joblib' else original(p))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=20).run()
    app.selectbox[0].select('TF-IDF baseline').run()
    return app


def test_single_result_and_stale_input(monkeypatch):
    app=open_app(monkeypatch)
    assert not app.exception
    next(b for b in app.button if b.label=='Analyze text →').click().run()
    assert any('Enter some' in w.value for w in app.warning)
    next(b for b in app.button if b.label=='A happy customer').click().run()
    next(b for b in app.button if b.label=='Analyze text →').click().run()
    assert not app.exception
    assert any(m.value=='Positive' for m in app.metric)
    app.text_area(key='single_text').input('نص مختلف').run()
    assert not any(m.label=='Sentiment' for m in app.metric)


def test_batch_limits_and_results(monkeypatch):
    app=open_app(monkeypatch)
    app.text_area(key='batch_text').input('\n'.join(['نص']*17))
    next(b for b in app.button if b.label=='Analyze collection →').click().run()
    assert any('at most 16' in w.value for w in app.warning)
    app.text_area(key='batch_text').input('خدمة ممتازة\nتوصيل سريع')
    next(b for b in app.button if b.label=='Analyze collection →').click().run()
    assert not app.exception
    assert any(m.label=='Texts analyzed' and m.value=='2' for m in app.metric)


def test_fresh_host_prepares_baseline(monkeypatch):
    original = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda p: False if p.name == 'baseline.joblib' else original(p))
    calls = []
    monkeypatch.setattr(maqsad.baseline, 'train_baseline', lambda **kwargs: calls.append(kwargs))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
    assert any('Hugging Face' in info.value for info in app.info)
    next(b for b in app.button if b.label == 'Prepare baseline model').click().run()
    assert not app.exception
    assert len(calls) == 1
    assert calls[0]['output'].endswith('artifacts/baseline.joblib')
    assert any('Baseline ready' in success.value for success in app.success)
