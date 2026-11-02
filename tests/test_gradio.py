import importlib.util
from pathlib import Path
import pytest

pytest.importorskip('gradio')
pytest.importorskip('spaces')
spec = importlib.util.spec_from_file_location('maqsad_gradio', Path(__file__).resolve().parents[1] / 'gradio_app.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


def test_interface_builds_without_download(monkeypatch):
    monkeypatch.setattr(ui, 'model_bundle', lambda: pytest.fail('Unexpected model download'))
    demo = ui.build_demo()
    assert demo.config['title'] == 'Maqsad · Arabic Language Studio'
    assert len(demo.config['dependencies']) >= 3
    demo.close()


def test_single_and_batch(monkeypatch):
    monkeypatch.setattr(ui, 'model_bundle', lambda: None)
    monkeypatch.setattr(ui, 'classify', lambda texts: [{'sarcasm': {'Sarcastic': 0.8, 'Not sarcastic': 0.2}, 'sentiment': {'Negative': 0.9, 'Neutral': 0.05, 'Positive': 0.05}} for _ in texts])
    assert ui.analyze('نص')[0]['Sarcastic'] == 0.8
    rows = ui.analyze_batch('نص أول\nنص ثان')
    assert len(rows) == 2
    assert rows[0][1:] == ['Sarcastic', 'Negative', 0.8]
    with pytest.raises(ui.gr.Error):
        ui.analyze(' ')
    with pytest.raises(ui.gr.Error):
        ui.analyze_batch('\n'.join(['نص'] * 17))
