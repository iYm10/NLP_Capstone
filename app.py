"""Maqsad's interactive Arabic language analysis studio."""
import json
import os
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from maqsad.baseline import predict

st.set_page_config(page_title='Maqsad · Language Studio', page_icon='✦', layout='wide')
st.markdown('''<style>
.block-container {max-width:1180px;padding-top:2.5rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {background:linear-gradient(115deg,#123849,#145e64);border-radius:24px;padding:38px 40px;color:#fff;margin-bottom:24px;}
.hero h1 {color:#fff;font-size:3rem;margin:8px 0;}
.hero p {color:#d4eeee;font-size:1.05rem;max-width:650px;}
.eyebrow {font-size:.75rem;letter-spacing:.2em;text-transform:uppercase;color:#9ee7d9;}
.pill {display:inline-block;border:1px solid #69aba8;border-radius:20px;padding:5px 12px;font-size:.75rem;color:#ddf6ef;margin-top:10px;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dfe8ec;border-radius:16px;padding:18px;}
[data-testid="stTextArea"] textarea {direction:rtl;text-align:right;font-size:1.1rem;line-height:1.9;}
[data-testid="stTabs"] button {font-weight:600;}
@media(max-width:600px){.hero{padding:24px}.hero h1{font-size:2.2rem}}
</style>''', unsafe_allow_html=True)

@st.cache_data
def benchmarks():
    records = []
    for label, path in [('TF-IDF baseline', ROOT/'reports/baseline_metrics.json'),
                        ('MARBERT · sentiment', ROOT/'reports/colab/marbert-sentiment-metrics.json'),
                        ('MARBERT · joint', ROOT/'reports/colab/marbert-joint-metrics.json')]:
        if path.exists():
            data = json.loads(path.read_text())['test']
            records.append({'Model':label,'Sentiment macro-F1':data['sentiment_macro_f1'],
                            'Sarcasm F1':data.get('sarcasm_f1'), 'Test examples':data['n']})
    return pd.DataFrame(records)

checkpoint = Path(os.environ.get('MAQSAD_CHECKPOINT', str(ROOT/'artifacts/marbert-joint')))
required_files = ['weights.pt', 'metrics.json', 'config.json', 'tokenizer_config.json']
missing_files = [name for name in required_files if not (checkpoint/name).is_file()]
ready_joint = not missing_files
options = ['MARBERT joint', 'TF-IDF baseline']

@st.cache_resource
def hosted_checkpoint(local_path, repo_id, revision):
    from maqsad.checkpoint import resolve_checkpoint
    return resolve_checkpoint(local_path, repo_id, revision)
with st.sidebar:
    st.markdown('## ✦ Maqsad')
    st.caption('ARABIC LANGUAGE STUDIO')
    st.divider()
    engine = st.selectbox('Analysis model', options)
    st.caption('The selected model is used for both single and batch analysis.')
    if not ready_joint:
        st.info('MARBERT will download from Hugging Face on first analysis. The first request may take several minutes.')
    if not (ROOT/'artifacts/baseline.joblib').exists():
        st.caption('Prepare the lightweight baseline to try the app while the neural checkpoint is unavailable.')
        if st.button('Prepare baseline model'):
            try:
                from maqsad.baseline import train_baseline
                with st.spinner('Downloading verified data and training the baseline. This may take several minutes…'):
                    train_baseline(data_dir=str(ROOT/'data/raw'), output=str(ROOT/'artifacts/baseline.joblib'), report_dir=str(ROOT/'artifacts/baseline-run'))
                st.success('Baseline ready. You can now analyze text.')
            except Exception as error:
                st.error(f'Baseline preparation failed: {error}')
    st.divider()
    st.markdown('**How to read the results**')
    st.caption('Sarcasm and sentiment are separate predictions. A positive phrase can express a negative opinion.')
    st.caption('Scores are uncalibrated model outputs, not certainty or judgments about a person.')
    with st.expander('Connect your trained model'):
        st.write('Hosted model: yahya-ai10/maqsad-marbert. A complete local checkpoint takes priority. Configure MAQSAD_CHECKPOINT, MAQSAD_MODEL_REPO or MAQSAD_MODEL_REVISION to override the defaults.')

st.markdown('''<div class="hero"><div class="eyebrow">WORDS · TONE · MEANING</div><h1>Read between the lines.</h1><p>Explore sarcasm and sentiment in Arabic. Try a phrase, compare a collection, and discover what the model sees.</p><span class="pill">Maqsad / Arabic Language Studio</span></div>''',unsafe_allow_html=True)


def analyze(texts):
    if engine == 'MARBERT joint':
        from maqsad.inference import predict_transformer
        resolved = hosted_checkpoint(str(checkpoint), os.environ.get('MAQSAD_MODEL_REPO', 'yahya-ai10/maqsad-marbert'), os.environ.get('MAQSAD_MODEL_REVISION', 'main'))
        return predict_transformer(texts, resolved)
    path = ROOT/'artifacts/baseline.joblib'
    if not path.exists():
        raise FileNotFoundError('Baseline model missing. Run: python -m maqsad.baseline')
    return predict(texts, str(path))


def set_example(text):
    st.session_state['single_text'] = text
    st.session_state.pop('single_result', None)


def csv_download(frame, label, key):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'), file_name='maqsad-analysis.csv', mime='text/csv', key=key)

playground, collection, research = st.tabs(['✦ Playground', '▤ Batch studio', '◈ Benchmarks'])
with playground:
    entry, help_col = st.columns([2,1], gap='large')
    with entry:
        st.subheader('What is the text really saying?')
        st.caption('Choose an example or enter your own Arabic text.')
        buttons = st.columns(3)
        examples=[('A sarcastic complaint','يا سلام على الخدمة، ساعتين وأنا أنتظر!'),('A happy customer','الخدمة ممتازة والتوصيل سريع'),('A simple update','وصل الطلب اليوم')]
        for col,(label,value) in zip(buttons,examples):
            col.button(label, on_click=set_example, args=(value,), width='stretch')
        text = st.text_area('Arabic text', key='single_text', height=180, max_chars=3000, placeholder='اكتب نصًا عربيًا هنا…')
        st.caption(f'{len(text)} characters · {len(text.split())} words')
        if st.button('Analyze text →', type='primary', width='stretch'):
            if not text.strip():
                st.warning('Enter some Arabic text first.')
            else:
                try:
                    with st.spinner('Reading the tone…'):
                        st.session_state['single_result']={'text':text,'engine':engine,**analyze([text])[0]}
                except (OSError, ValueError, ImportError, KeyError, RuntimeError) as error:
                    st.error(f'Analysis could not complete: {error}')
    with help_col:
        st.subheader('Two perspectives')
        st.markdown('**Tone**\n\nDoes the wording suggest sarcasm?\n\n**Sentiment**\n\nIs the expressed attitude negative, neutral, or positive?')
        st.info('Examples are illustrative inputs, not guaranteed correct predictions. Context can change the meaning.')
    result = st.session_state.get('single_result')
    if result and result['text']==text and result['engine']==engine:
        st.divider()
        st.caption(f"ANALYSIS COMPLETE · {engine}")
        a,b,c = st.columns(3)
        a.metric('Tone', 'Sarcastic' if result['sarcasm'] else 'Not sarcastic')
        b.metric('Sentiment', result['sentiment'].title())
        c.metric('Sarcasm score', f"{result['sarcasm_score']:.1%}")
        st.progress(float(result['sarcasm_score']),text='Model score for the sarcastic class')
        st.caption(f"Predicted sentiment score: {result['sentiment_score']:.1%}. Scores are not calibrated confidence estimates.")
        csv_download(pd.DataFrame([result]),'Download this analysis','single_csv')
    elif result:
        st.caption('Text or model changed. Analyze again to refresh the result.')

with collection:
    st.subheader('Spot patterns across a collection')
    st.caption('Paste one Arabic text per line. Up to 16 texts per analysis, 3,000 characters per text.')
    batch = st.text_area('Texts to analyze',height=200,key='batch_text')
    lines = [line.strip() for line in batch.splitlines() if line.strip()]
    if st.button('Analyze collection →',type='primary'):
        if not lines:
            st.warning('Enter at least one text.')
        elif len(lines)>16 or any(len(line)>3000 for line in lines):
            st.warning('Use at most 16 texts, each no longer than 3,000 characters.')
        else:
            try:
                with st.spinner('Analyzing the collection…'):
                    records=[{'text':t,'engine':engine,**p} for t,p in zip(lines,analyze(lines))]
                    st.session_state['batch_result']={'lines':lines,'engine':engine,'records':records}
            except (OSError,ValueError,ImportError,KeyError,RuntimeError) as error:
                st.error(f'Analysis could not complete: {error}')
    saved=st.session_state.get('batch_result')
    if saved and saved['lines']==lines and saved['engine']==engine:
        frame=pd.DataFrame(saved['records'])
        a,b=st.columns(2)
        a.metric('Texts analyzed',len(frame))
        b.metric('Predicted sarcastic',int(frame.sarcasm.sum()))
        st.bar_chart(frame.sentiment.value_counts().rename('Texts'),color='#0d9488')
        st.dataframe(frame,hide_index=True,width='stretch')
        csv_download(frame,'Download collection CSV','batch_csv')

with research:
    st.subheader('Evidence behind the experiment')
    st.caption('Official 3,000-example test split · seed 42 · validation-selected checkpoints')
    frame=benchmarks()
    st.bar_chart(frame.set_index('Model')[['Sentiment macro-F1','Sarcasm F1']],horizontal=True,color=['#0d9488','#7371c7'])
    st.dataframe(frame.style.format({'Sentiment macro-F1':'{:.2%}','Sarcasm F1':'{:.2%}'},na_rep='Not trained'),hide_index=True,width='stretch')
    st.info('Joint training did not improve sentiment in this run: 65.57% versus 65.71% macro-F1. Multiple seeds are needed before drawing a robust conclusion.')
    st.caption('Neural metrics come from the supplied Colab reports; local baseline metrics were reproduced here. Metrics do not imply that neural weights are installed in this app. Missing sarcasm scores indicate an untrained task, not zero performance.')
    with st.expander('Dialect coverage and limitations'):
        st.write('The dataset contains historical tweets and uneven dialect representation. The Maghrebi test group has only two examples; its score is not a reliable regional estimate. Gulf labels cover several countries. Single tweets may lack the context needed to interpret sarcasm.')
st.divider()
st.caption('MAQSAD · An Arabic NLP research project · Predictions support exploration, not definitive judgments.')
