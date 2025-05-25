# ------------------------------
# Load Pretrained Models & Tokenizers
# ------------------------------

import os
import json
import gdown
import torch
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoTokenizer, TFAutoModel, PegasusTokenizer, PegasusForConditionalGeneration

# Google Drive file IDs for required resources
GOOGLE_DRIVE_FILES = {
    "bilstm_model.h5":     "1Js-MS8K8kWT23STTrYo9WizRpdr2qcYL",
    "cnn_model.h5":        "1mUgDlby6p4c6UF7wph85nSd0Gj_NLy0B",
    "bert_model.h5":       "1l2wbEES9-VaOGSVNEmK3bUqH3o2VVaXr",
    "tokenizer.json":      "1Cgn06cl2D-TG4wAintOL_Ng3gCir28Rv",
}

def download_from_drive(filename, file_id):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists, skipping.")

@st.cache_resource
def load_all_models():
    # Download files if missing
    for filename, file_id in GOOGLE_DRIVE_FILES.items():
        download_from_drive(filename, file_id)

    # Load PEGASUS (for headline generation)
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    # Load BERT tokenizer and model
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_base = TFAutoModel.from_pretrained("bert-base-uncased")

    input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
    outputs = bert_base(input_ids, attention_mask=attention_mask)
    cls_output = outputs.last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3)(cls_output)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(4, activation='softmax')(x)
    bert_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    bert_model.load_weights("bert_model.h5")

    # Load BiLSTM and CNN models
    bilstm_model = load_model("bilstm_model.h5")
    cnn_model = load_model("cnn_model.h5")

    # Load tokenizer
    with open("tokenizer.json") as f:
        tokenizer = tokenizer_from_json(json.load(f))

    return pegasus_tokenizer, pegasus_model, bert_tokenizer, bert_model, bilstm_model, cnn_model, tokenizer


pegasus_tokenizer, pegasus_model, bert_tokenizer, bert_model, bilstm_model, cnn_model, tokenizer = load_all_models()
max_len = 128  # or use your actual padded length

# ------------------------------
# Utilities
# ------------------------------

def classify_article(article):
    # Preprocess
    seq = tokenizer.texts_to_sequences([article])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # BiLSTM and CNN
    bilstm_probs = bilstm_model.predict(padded_seq)
    cnn_probs = cnn_model.predict(padded_seq)

    # BERT
    encod = bert_tokenizer(article, truncation=True, padding='max_length', max_length=128, return_tensors='tf')
    bert_probs = bert_model.predict({'input_ids': encod['input_ids'], 'attention_mask': encod['attention_mask']})

    confidences = [np.max(bilstm_probs), np.max(cnn_probs), np.max(bert_probs)]
    preds = [np.argmax(bilstm_probs), np.argmax(cnn_probs), tf.argmax(bert_probs, axis=1).numpy()[0]]
    model_names = ['BiLSTM', 'CNN', 'BERT']
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    best_idx = np.argmax(confidences)

    return {
        'best_model': model_names[best_idx],
        'predicted_class': class_names[preds[best_idx]],
        'confidence': confidences[best_idx],
        'all_results': dict(zip(model_names, zip([class_names[i] for i in preds], confidences)))
    }

def generate_headline(article):
    inputs = pegasus_tokenizer(article, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    with torch.no_grad():
        summary_ids = pegasus_model.generate(inputs["input_ids"], num_beams=4, max_length=60, early_stopping=True)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="📰 Smart News Analyzer", layout="wide")
st.title("📰 Smart News Analyzer")
st.markdown("Classify a news article and generate a headline using BiLSTM, CNN, BERT and PEGASUS!")

article_input = st.text_area("Enter a news article (short paragraph)", height=200)

if st.button("Analyze"):
    if not article_input.strip():
        st.warning("Please enter a news article.")
    else:
        with st.spinner("Analyzing..."):
            classification = classify_article(article_input)
            headline = generate_headline(article_input)

        # Show classification
        st.subheader("📌 Classification Results")
        st.success(f"**Category:** `{classification['predicted_class']}`")
        st.info(f"**Best Model:** `{classification['best_model']}` with confidence `{classification['confidence']:.2f}`")

        with st.expander("🔍 View all model predictions"):
            for model, (pred, conf) in classification['all_results'].items():
                st.write(f"**{model}** → {pred} ({conf:.2f})")

        # Show headline
        st.subheader("📰 Generated Headline")
        st.write(f"**{headline}**")
