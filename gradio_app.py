"""Maqsad Gradio application for CPU and Hugging Face ZeroGPU."""
import os
import csv
import html
import tempfile
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


EMPTY_RESULT = '<div class="empty-result"><span>✦</span><h3>A little context. A clearer picture.</h3><p>Your analysis will appear here. Start with an example or write your own Arabic text.</p></div>'


def text_stats(text):
    return f"{len(text or ''):,} / 3,000 characters · {len((text or '').split())} words"


def analyze_view(text):
    tone, sentiment = analyze(text)
    tone_name = max(tone, key=tone.get)
    sentiment_name = max(sentiment, key=sentiment.get)
    score = tone['Sarcastic']
    summary = f'''<div class="result-grid">
        <div class="result-card"><small>DETECTED TONE</small><h2>{tone_name}</h2><p>Sarcasm score · {score:.1%}</p></div>
        <div class="result-card accent"><small>EXPRESSED SENTIMENT</small><h2>{sentiment_name}</h2><p>Predicted-class score · {sentiment[sentiment_name]:.1%}</p></div>
        </div><div class="analyzed-text"><small>ANALYZED TEXT</small><p dir="rtl">{html.escape(text)}</p></div>'''
    return summary, tone, sentiment


def csv_safe(value):
    # Prevent spreadsheet formulas when exporting user-authored text.
    value = str(value)
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value


def batch_view(text):
    rows = analyze_batch(text)
    sarcastic = sum(row[1] == 'Sarcastic' for row in rows)
    negative = sum(row[2] == 'Negative' for row in rows)
    summary = f"**{len(rows)} texts analyzed** · {sarcastic} predicted sarcastic · {negative} predicted negative"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix='maqsad-', encoding='utf-8-sig', newline='', delete=False) as output:
        writer = csv.writer(output)
        writer.writerow(['Text', 'Tone', 'Sentiment', 'Sarcasm score'])
        for row in rows:
            writer.writerow([csv_safe(row[0]), *row[1:]])
        name = output.name
    return rows, summary, name


CSS = '''
.gradio-container {max-width:1160px!important;margin:auto!important;padding:24px!important;}
#hero {position:relative;overflow:hidden;padding:40px;border-radius:26px;background:linear-gradient(120deg,#102f40,#14656a);margin-bottom:22px;}
#hero:after {content:"✦";position:absolute;right:40px;top:-25px;font-size:240px;color:#ffffff0b;pointer-events:none;}
#hero h1 {color:#fff!important;font-size:clamp(2rem,5vw,3.5rem);letter-spacing:-.055em;line-height:1.15;margin:14px 0;}
#hero p {color:#d1e9e8!important;max-width:620px;font-size:16px;line-height:1.7;}
.eyebrow {color:#a0ecda;font-size:11px;letter-spacing:.2em;font-weight:700;}
.hero-tags {display:flex;gap:8px;flex-wrap:wrap;margin-top:24px;}
.hero-tags span {color:#defffa;border:1px solid #ffffff35;border-radius:99px;font-size:11px;padding:5px 12px;}
#arabic-input textarea,#batch-input textarea {direction:rtl;text-align:right;font-size:19px;line-height:1.9;}
#input-panel {padding:20px;border-radius:20px;border:1px solid var(--border-color-primary);}
.empty-result {padding:35px 22px;border:1px dashed var(--border-color-primary);border-radius:20px;text-align:center;min-height:215px;}
.empty-result span {font-size:40px;color:#0d9488;}
.empty-result h3 {font-size:20px;margin:12px 0;}
.empty-result p {color:var(--body-text-color-subdued);line-height:1.6;}
.result-grid {display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.result-card {padding:22px 18px;border-radius:18px;background:#e5f5f1;color:#153f3d;border:1px solid #c6e6dd;}
.result-card.accent {background:#eeecfc;color:#403964;border-color:#dcd7f3;}
.result-card small,.analyzed-text small {font-size:10px;letter-spacing:.13em;font-weight:700;}
.result-card h2 {font-size:24px;margin:12px 0!important;color:inherit!important;}
.result-card p {font-size:12px;color:inherit!important;}
.analyzed-text {padding:18px 4px;overflow-wrap:anywhere;}
.analyzed-text p {font-size:17px;line-height:1.8;}
#counter {font-size:12px;color:var(--body-text-color-subdued);}
#footer {text-align:center;font-size:12px;margin-top:28px;color:var(--body-text-color-subdued);}
@media(max-width:600px){.gradio-container{padding:12px!important}#hero{padding:26px}.result-grid{grid-template-columns:1fr}#hero:after{display:none}}
'''


def build_demo():
    with gr.Blocks(title='Maqsad · Arabic Language Studio', theme=gr.themes.Soft(primary_hue='teal', secondary_hue='violet'), css=CSS, delete_cache=(3600, 3600)) as demo:
        gr.HTML('''<div id="hero"><div class="eyebrow">MAQSAD / ARABIC LANGUAGE STUDIO</div><h1>Words say one thing.<br>Tone tells the rest.</h1><p>Explore the sentiment behind Arabic expressions with a model trained to recognize both emotion and sarcasm.</p><div class="hero-tags"><span>Arabic NLP</span><span>MARBERT</span><span>Sarcasm + Sentiment</span></div></div>''')
        with gr.Tab('✦ Playground'):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_id='input-panel'):
                    gr.Markdown('### Start with a phrase\nChoose an example, or write something of your own.')
                    text = gr.Textbox(label='Arabic text', placeholder='اكتب نصًا عربيًا هنا…', lines=5, max_lines=10, elem_id='arabic-input')
                    counter = gr.Markdown(text_stats(''), elem_id='counter')
                    with gr.Row():
                        ex1 = gr.Button('Sarcastic complaint', size='sm')
                        ex2 = gr.Button('Happy customer', size='sm')
                        ex3 = gr.Button('Simple update', size='sm')
                    with gr.Row():
                        submit = gr.Button('Analyze text →', variant='primary', scale=3)
                        clear = gr.ClearButton([text], value='Reset', scale=1)
                    gr.Markdown('First analysis may take a few minutes to load the model. Example labels describe the intended wording, not guaranteed predictions.')
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown('### Your analysis\nTwo perspectives on the same expression.')
                    summary = gr.HTML(EMPTY_RESULT)
                    with gr.Accordion('Explore model scores', open=False):
                        tone = gr.Label(label='Tone distribution', num_top_classes=2)
                        sentiment = gr.Label(label='Sentiment distribution', num_top_classes=3)
                    gr.Markdown('Scores are uncalibrated model outputs. Sarcasm can depend on context that a single sentence does not provide.')
            for button, phrase in [(ex1, 'يا سلام على الخدمة، ساعتين وأنا أنتظر!'), (ex2, 'الخدمة ممتازة والتوصيل سريع'), (ex3, 'وصل الطلب اليوم')]:
                button.click(lambda value=phrase: value, outputs=text, queue=False)
            text.change(lambda value: (text_stats(value), EMPTY_RESULT, None, None), text, [counter, summary, tone, sentiment], queue=False)
            clear.add([tone, sentiment])
            submit.click(analyze_view, text, [summary, tone, sentiment], concurrency_id='model', concurrency_limit=1)
        with gr.Tab('▤ Batch studio'):
            gr.Markdown('### From individual opinions to a bigger picture\nAnalyze up to 16 texts at once. Add one text per line, with a maximum of 3,000 characters per text.')
            batch = gr.Textbox(label='Arabic collection', lines=7, elem_id='batch-input')
            with gr.Row():
                run = gr.Button('Analyze collection →', variant='primary')
                reset_batch = gr.ClearButton([batch], value='Reset collection')
            batch_summary = gr.Markdown('Your collection summary will appear after analysis.')
            table = gr.Dataframe(headers=['Text', 'Tone', 'Sentiment', 'Sarcasm score'], datatype=['str','str','str','number'], interactive=False, label='Results', wrap=True)
            download = gr.File(label='Download results · CSV', interactive=False)
            batch.change(lambda: ([], 'Collection changed. Analyze to refresh results.', None), outputs=[table, batch_summary, download], queue=False)
            reset_batch.add([table, download])
            run.click(batch_view, batch, [table, batch_summary, download], concurrency_id='model', concurrency_limit=1)
        with gr.Tab('◈ Research'):
            gr.Markdown('''## Built around a research question
Does learning sarcasm alongside sentiment improve Arabic sentiment classification?

| Model | Sentiment macro-F1 | Sarcasm F1 |
|---|---:|---:|
| TF-IDF baseline | 56.75% | 54.04% |
| MARBERT sentiment-only | **65.71%** | — |
| MARBERT joint | 65.57% | **58.56%** |

**What the results show:** MARBERT improves over the baseline, but joint learning does not improve sentiment over the sentiment-only model in this run.

ArSarcasm-v2 · 3,000 test examples · seed 42. Scores are held-out evaluation results, not predictions for the text entered above.''')
            with gr.Accordion('Data, limitations and methodology'):
                gr.Markdown('Historical Arabic tweets cover multiple dialects with unequal representation. The Maghrebi test group has only two examples. Results need further evaluation across seeds and domains. [Read the experiment report](https://github.com/iYm10/NLP_Capstone/blob/main/reports/EXPERIMENT_REPORT.md).')
        gr.Markdown('[GitHub](https://github.com/iYm10/NLP_Capstone) · [Model on Hugging Face](https://huggingface.co/yahya-ai10/maqsad-marbert) · [Streamlit dashboard](https://nlpcapstone-ya.streamlit.app/)\n\nMAQSAD · Arabic language, with context in mind.', elem_id='footer')
    return demo.queue(default_concurrency_limit=1, max_size=20)


if __name__ == '__main__':
    build_demo().launch()
