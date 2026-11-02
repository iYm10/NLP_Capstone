"""Resolve a complete local checkpoint or download the hosted artifact."""
from pathlib import Path

REQUIRED = ('weights.pt', 'metrics.json', 'config.json', 'tokenizer_config.json')
DEFAULT_REPO = 'yahya-ai10/maqsad-marbert'


def validate_checkpoint(path):
    path = Path(path)
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if not any((path / name).is_file() for name in ('tokenizer.json', 'vocab.txt')):
        missing.append('tokenizer.json or vocab.txt')
    if missing:
        raise FileNotFoundError('Incomplete checkpoint: ' + ', '.join(missing))
    return path


def resolve_checkpoint(local_path, repo_id=DEFAULT_REPO, revision='main'):
    try:
        return validate_checkpoint(local_path)
    except FileNotFoundError:
        pass
    from huggingface_hub import snapshot_download
    # Downloads are cached by immutable resolved commit by the Hub client.
    downloaded = snapshot_download(repo_id=repo_id, revision=revision,
        allow_patterns=['weights.pt', '*.json', 'vocab.txt', '*.model'])
    return validate_checkpoint(downloaded)
