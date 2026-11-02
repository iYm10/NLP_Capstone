"""Maqsad Gradio application for CPU and Hugging Face ZeroGPU."""
import os
import sys
from pathlib import Path
from functools import lru_cache

# Import Spaces before torch for ZeroGPU runtime integration.
import spaces
import torch
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from maqsad.checkpoint import resolve_checkpoint
from maqsad.inference import load_checkpoint
from maqsad.data import normalize


@lru_cache(maxsize=1)
def model_bundle():
    path = resolve_checkpoint(
        os.environ.get('MAQSAD_CHECKPOINT', str(ROOT / 'artifacts/marbert-joint')),
        os.environ.get('MAQSAD_MODEL_REPO', 'yahya-ai10/maqsad-marbert'),
        os.environ.get('MAQSAD_MODEL_REVISION', 'main'),
    )
    return load_checkpoint(str(path.resolve()))


@spaces.GPU(duration=60)
def classify(texts):
    config, tokenizer, model = model_bundle()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    try:
        tokens = tokenizer([normalize(text) for text in texts], padding=True,
                           truncation=True, max_length=config['max_length'], return_tensors='pt')
        with torch.inference_mode():
            logits = model(**{k: v.to(device) for k, v in tokens.items()})
        sarcasm = logits['sarcasm'].softmax(-1).cpu().tolist()
        sentiment = logits['sentiment'].softmax(-1).cpu().tolist()
        return [{'sarcasm': {'Not sarcastic': s[0], 'Sarcastic': s[1]},
                 'sentiment': dict(zip(['Negative', 'Neutral', 'Positive'], m))}
                for s, m in zip(sarcasm, sentiment)]
    finally:
        model.to('cpu')


def analyze(text):
    if not text or not text.strip():
        raise gr.Error('Enter Arabic text to begin.')
    if len(text) > 3000:
        raise gr.Error('Please use 3,000 characters or fewer.')
    # Download and load on CPU before requesting the timed GPU allocation.
    model_bundle()
    result = classify([text])[0]
    return result['sarcasm'], result['sentiment']


def analyze_batch(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) > 16 or any(len(line) > 3000 for line in lines):
        raise gr.Error('Enter 1–16 texts, one per line, with at most 3,000 characters each.')
    model_bundle()
    results = classify(lines)
    return [[line, max(r['sarcasm'], key=r['sarcasm'].get),
             max(r['sentiment'], key=r['sentiment'].get), round(r['sarcasm']['Sarcastic'], 4)]
            for line, r in zip(lines, results)]


CSS = '''
.gradio-container {max-width:1100px!important;margin:auto!important;}
#hero {padding:30px;border-radius:22px;background:linear-gradient(115deg,#123849,#145e64);margin-bottom:16px;}
#hero h1,#hero p {color:white!important;}
#hero h1 {font-size:2.6rem;letter-spacing:-.04em;}
#arabic-input textarea,#batch-input textarea {direction:rtl;text-align:right;font-size:18px;line-height:1.8;}
'''


def build_demo():
    with gr.Blocks(title='Maqsad · Arabic Language Studio', theme=gr.themes.Soft(primary_hue='teal', secondary_hue='violet'), css=CSS) as demo:
        gr.Markdown('# Maqsad\nRead between the lines. Explore sarcasm and sentiment in Arabic.', elem_id='hero')
        with gr.Tab('Playground'):
            with gr.Row():
                with gr.Column(scale=3):
                    text = gr.Textbox(label='Arabic text', placeholder='اكتب نصًا عربيًا هنا…', lines=5, max_lines=10, elem_id='arabic-input')
                    gr.Examples(examples=[['يا سلام على الخدمة، ساعتين وأنا أنتظر!'], ['الخدمة ممتازة والتوصيل سريع'], ['وصل الطلب اليوم']], inputs=text)
                    with gr.Row():
                        submit = gr.Button('Analyze text →', variant='primary')
                        clear = gr.ClearButton([text])
                with gr.Column(scale=2):
                    tone = gr.Label(label='Tone', num_top_classes=2)
                    sentiment = gr.Label(label='Sentiment', num_top_classes=3)
                    gr.Markdown('Scores are uncalibrated model outputs. Sarcasm can depend on context beyond a single sentence.')
            clear.add([tone, sentiment])
            submit.click(analyze, text, [tone, sentiment], concurrency_id='model', concurrency_limit=1)
        with gr.Tab('Batch studio'):
            batch = gr.Textbox(label='One Arabic text per line', lines=7, elem_id='batch-input')
            run = gr.Button('Analyze collection →', variant='primary')
            table = gr.Dataframe(headers=['Text', 'Tone', 'Sentiment', 'Sarcasm score'], datatype=['str','str','str','number'], interactive=False, label='Results')
            run.click(analyze_batch, batch, table, concurrency_id='model', concurrency_limit=1)
        with gr.Tab('Research'):
            gr.Markdown('''## Experiment results
| Model | Sentiment macro-F1 | Sarcasm F1 |
|---|---:|---:|
| TF-IDF | 56.75% | 54.04% |
| MARBERT sentiment-only | 65.71% | — |
| MARBERT joint | 65.57% | 58.56% |

ArSarcasm-v2 · 3,000 test examples · seed 42. Joint learning did not improve sentiment in this run. Dialect coverage is uneven, and the results require further evaluation across seeds and domains.

[Source code](https://github.com/iYm10/NLP_Capstone) · [Model](https://huggingface.co/yahya-ai10/maqsad-marbert)''')
        gr.Markdown('The first analysis downloads the checkpoint and may take several minutes. Model predictions support exploration, not definitive judgments about a person.')
    return demo.queue(default_concurrency_limit=1, max_size=20)


if __name__ == '__main__':
    build_demo().launch()
