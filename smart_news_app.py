# -------------------- Imports and Configuration --------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import logging
import asyncio
import atexit
import warnings

import gdown
import numpy as np
import tensorflow as tf
import torch

import transformers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoTokenizer, TFAutoModel, PegasusTokenizer, PegasusForConditionalGeneration

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="📰 Smart News Analyzer", layout="wide")
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.basicConfig(filename="app_debug.log", level=logging.ERROR)

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

def clean_shutdown():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
    except Exception:
        pass
atexit.register(clean_shutdown)

# -------------------- Constants --------------------
GOOGLE_DRIVE_FILES = {
    "bilstm_model.h5": "1Js-MS8K8kWT23STTrYo9WizRpdr2qcYL",
    "cnn_model.h5": "1mUgDlby6p4c6UF7wph85nSd0Gj_NLy0B",
    "bert_model.h5": "1l2wbEES9-VaOGSVNEmK3bUqH3o2VVaXr",
    "tokenizer.json": "1Cgn06cl2D-TG4wAintOL_Ng3gCir28Rv",
}
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
max_len = 128

# -------------------- File Download --------------------
@st.cache_resource(show_spinner=True)
def download_model_files():
    for filename, file_id in GOOGLE_DRIVE_FILES.items():
        if not os.path.exists(filename):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)

download_model_files()

# -------------------- Custom LSTM Fix --------------------
from tensorflow.keras.layers import LSTM as KerasLSTM
from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class CustomLSTM(KerasLSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

tf.keras.layers.LSTM = CustomLSTM

# -------------------- Model Caches --------------------
models = {
    'pegasus_tokenizer': None,
    'pegasus_model': None,
    'bert_tokenizer': None,
    'bert_model': None,
    'bilstm_model': None,
    'cnn_model': None,
    'tokenizer': None,
}

# -------------------- Loaders --------------------
def load_pegasus():
    """Load PEGASUS tokenizer and model once."""
    if models['pegasus_model'] is None or models['pegasus_tokenizer'] is None:
        models['pegasus_tokenizer'] = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        models['pegasus_model'] = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

def load_bert():
    """Load BERT tokenizer and model with pretrained weights, cache in models dict."""
    if models['bert_model'] is not None and models['bert_tokenizer'] is not None:
        return  # Already loaded

    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        base = TFAutoModel.from_pretrained("bert-base-uncased")
        base.trainable = False  # Freeze BERT base layers

        input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

        outputs = base(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        x = tf.keras.layers.Dropout(0.3)(cls_output)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=out)
        model.load_weights("bert_model.h5")

        models['bert_tokenizer'] = tokenizer
        models['bert_model'] = model
    except Exception as e:
        logging.exception("Error loading BERT model")
        models['bert_model'] = None
        models['bert_tokenizer'] = None

def load_bilstm_cnn_tokenizer():
    """Load BiLSTM, CNN models and tokenizer once."""
    if models['bilstm_model'] is None:
        try:
            models['bilstm_model'] = load_model("bilstm_model.h5", custom_objects={"LSTM": CustomLSTM})
        except Exception as e:
            logging.exception("Error loading BiLSTM model")
            models['bilstm_model'] = None
    if models['cnn_model'] is None:
        try:
            models['cnn_model'] = load_model("cnn_model.h5")
        except Exception as e:
            logging.exception("Error loading CNN model")
            models['cnn_model'] = None
    if models['tokenizer'] is None:
        try:
            with open("tokenizer.json", "r") as f:
                models['tokenizer'] = tokenizer_from_json(f.read())
        except Exception as e:
            logging.exception("Error loading tokenizer JSON")
            models['tokenizer'] = None

# -------------------- Prediction Logic --------------------
def classify_article(article, selected_models):
    """Classify article text using selected models and return best prediction."""
    try:
        results = {}

        if any(m in selected_models for m in ["BiLSTM", "CNN"]):
            load_bilstm_cnn_tokenizer()
            if models['tokenizer'] is None:
                logging.error("Tokenizer not loaded, cannot process BiLSTM/CNN")
            else:
                seq = models['tokenizer'].texts_to_sequences([article])
                padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

        if "BiLSTM" in selected_models and models['bilstm_model'] is not None:
            probs = models['bilstm_model'].predict(padded, verbose=0)
            results['BiLSTM'] = (np.argmax(probs), np.max(probs))

        if "CNN" in selected_models and models['cnn_model'] is not None:
            probs = models['cnn_model'].predict(padded, verbose=0)
            results['CNN'] = (np.argmax(probs), np.max(probs))

        if "BERT" in selected_models:
            load_bert()
            if models['bert_model'] is not None and models['bert_tokenizer'] is not None:
                tokenizer = models['bert_tokenizer']
                inputs = tokenizer(article, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')
                probs = models['bert_model'].predict(inputs, verbose=0)
                results['BERT'] = (np.argmax(probs), np.max(probs))
            else:
                logging.error("BERT model or tokenizer not loaded")

        if not results:
            return None

        best_model = max(results, key=lambda k: results[k][1])
        best_idx, best_conf = results[best_model]
        return {
            'best_model': best_model,
            'predicted_class': class_names[best_idx],
            'confidence': best_conf,
            'all_results': {m: (class_names[i], c) for m, (i, c) in results.items()}
        }

    except Exception as e:
        logging.exception("Classification error")
        return None

def generate_headline(article):
    """Generate a headline using PEGASUS summarization."""
    try:
        load_pegasus()
        tokenizer = models['pegasus_tokenizer']
        model = models['pegasus_model']
        inputs = tokenizer(article, return_tensors="pt", truncation=True, padding='max_length', max_length=256)
        with torch.no_grad():
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=60, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logging.exception("Headline generation error")
        return "[Headline Generation Failed]"

# -------------------- Streamlit UI --------------------
st.title("📰 Smart News Analyzer")
st.markdown("Classify a news article and generate a headline using **BiLSTM, CNN, BERT**, and **PEGASUS**!")

model_options = ["BiLSTM", "CNN", "BERT", "PEGASUS (Headline)"]
selected_models = st.multiselect("Select models to use:", model_options, default=model_options)

article_input = st.text_area("📝 Enter a news article:", height=200)

if st.button("🚀 Analyze"):
    if not article_input.strip():
        st.warning("Please enter a news article.")
    else:
        with st.spinner("Analyzing..."):
            classification = classify_article(article_input, selected_models)

            if classification:
                st.subheader("📊 Classification Results")
                st.write(f"**Best Model:** {classification['best_model']}")
                st.write(f"**Predicted Class:** {classification['predicted_class']}")
                st.write(f"**Confidence:** {classification['confidence']:.2f}")
                st.write("### All Model Results:")
                for m, (cls, conf) in classification['all_results'].items():
                    st.write(f"- {m}: {cls} ({conf:.2f})")
            else:
                st.error("Classification failed or no model selected.")

            if "PEGASUS (Headline)" in selected_models:
                headline = generate_headline(article_input)
                st.subheader("📰 Generated Headline")
                st.markdown(f"> {headline}")
