
import streamlit as st
import torch
import re
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from Preprocessor.preprocessor import preprocess
from docx import Document

# Cümle bölücü
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# Başlık
st.title("DOCX Üzerinden Türkçe Nefret Söylemi Tespiti")

# Model ve Tokenizer yükleme
@st.cache_resource
def load_model_and_tokenizer():
    peft_model = "VRLLab/TurkishBERTweet-Lora-HS"
    peft_config = PeftConfig.from_pretrained(peft_model)

    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path, padding_side="right"
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    id2label = {0: "No", 1: "Yes"}
    model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        num_labels=len(id2label),
        id2label=id2label
    )
    model = PeftModel.from_pretrained(model, peft_model)
    return model, tokenizer, id2label

model, tokenizer, id2label = load_model_and_tokenizer()

# DOCX yükleme
uploaded_docx = st.file_uploader("Bir .docx dosyası yükleyin", type=["docx"])
if uploaded_docx is not None:
    document = Document(uploaded_docx)
    full_text = "\n".join([para.text for para in document.paragraphs if para.text.strip()])
    sentences = split_sentences(full_text)
    preprocessed_sentences = [preprocess(s) for s in sentences]

    results = []
    hate_count = 0

    with torch.no_grad():
        for sent, preproc in zip(sentences, preprocessed_sentences):
            ids = tokenizer.encode_plus(preproc, return_tensors="pt", truncation=True)
            label_id = model(**ids).logits.argmax(-1).item()
            label = id2label[label_id]
            if label == "Yes":
                hate_count += 1
            results.append({"sentence": sent, "hate_speech": label})

    df = pd.DataFrame(results)
    st.subheader("Analiz Sonuçları")
    st.dataframe(df)

    ratio = hate_count / len(results) * 100
    st.subheader(f"Nefret Söylemi Oranı: %{ratio:.2f}")

    # CSV indirme bağlantısı
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Sonuçları CSV olarak indir",
        data=csv,
        file_name="hatespeech_results.csv",
        mime="text/csv",
    )
