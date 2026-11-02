from pathlib import Path
import pytest
from maqsad.checkpoint import resolve_checkpoint, validate_checkpoint, REQUIRED


def make_checkpoint(path):
    path.mkdir()
    for name in (*REQUIRED, 'vocab.txt'):
        (path / name).touch()
    return path


def test_local_checkpoint_needs_no_download(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kw: pytest.fail('Unexpected network call'))
    path = make_checkpoint(tmp_path / 'local')
    assert resolve_checkpoint(path) == path


def test_missing_weights_uses_hosted_checkpoint(tmp_path, monkeypatch):
    import huggingface_hub
    remote = make_checkpoint(tmp_path / 'cached')
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(remote)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    assert resolve_checkpoint(tmp_path / 'missing', revision='test-revision') == remote
    assert calls[0]['repo_id'] == 'yahya-ai10/maqsad-marbert'
    assert calls[0]['revision'] == 'test-revision'


def test_incomplete_download_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match='weights.pt'):
        validate_checkpoint(tmp_path)
