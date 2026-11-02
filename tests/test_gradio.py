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


def test_result_card_escapes_input(monkeypatch):
    monkeypatch.setattr(ui, 'analyze', lambda text: ({'Sarcastic': 0.8, 'Not sarcastic': 0.2}, {'Negative': 0.9, 'Neutral': 0.05, 'Positive': 0.05}))
    summary, tone, sentiment = ui.analyze_view('<script>alert(1)</script>')
    assert '<script>' not in summary
    assert '&lt;script&gt;' in summary
    assert '80.0%' in summary
    assert tone['Sarcastic'] == 0.8
    assert '2 words' in ui.text_stats('نص عربي')


def test_batch_export(monkeypatch):
    import csv
    monkeypatch.setattr(ui, 'analyze_batch', lambda text: [['=1+1', 'Sarcastic', 'Negative', 0.8]])
    rows, summary, filename = ui.batch_view('=1+1')
    try:
        with open(filename, encoding='utf-8-sig', newline='') as source:
            data = list(csv.reader(source))
        assert data[1][0] == "'=1+1"
        assert '1 texts analyzed' in summary
        assert len(rows) == 1
    finally:
        Path(filename).unlink()
