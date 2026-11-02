"""Publish a minimal Gradio bundle using an existing Hugging Face login."""
from pathlib import Path
import shutil
import tempfile
from huggingface_hub import HfApi


def main():
    root = Path(__file__).resolve().parents[1]
    api = HfApi()
    api.whoami()  # Requires a login with write access; never store tokens in source.
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory)
        shutil.copy2(root / 'gradio_app.py', bundle / 'gradio_app.py')
        for name in ['README.md', 'requirements.txt']:
            shutil.copy2(root / 'deployment/gradio' / name, bundle / name)
        shutil.copytree(root / 'src/maqsad', bundle / 'src/maqsad', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        api.upload_folder(repo_id='yahya-ai10/maqsad-demo', repo_type='space',
                          folder_path=str(bundle), commit_message='Deploy Maqsad Gradio interface')
    print('Deployment uploaded: https://huggingface.co/spaces/yahya-ai10/maqsad-demo')


if __name__ == '__main__':
    main()
