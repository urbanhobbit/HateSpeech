import streamlit as st
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from Preprocessor import preprocess
from docx import Document

# Cümle bölücü
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

st.set_page_config(page_title="Metin Analizi Aracı", layout="wide")
st.title("📊 Türkçe Metin Analizi Aracı")
st.markdown("Bu araç DOCX dosyanızdaki ya da doğrudan girdiğiniz her bir cümleyi **nefret söylemi** ve **duygu analizi** açısından inceler.")

# Model ve tokenizer yükleyici
@st.cache_resource
def load_models():
    # Hate Speech Model
    peft_hs = "VRLLab/TurkishBERTweet-Lora-HS"
    cfg_hs = PeftConfig.from_pretrained(peft_hs)
    tokenizer_hs = AutoTokenizer.from_pretrained(cfg_hs.base_model_name_or_path, padding_side="right")
    if tokenizer_hs.pad_token_id is None:
        tokenizer_hs.pad_token_id = tokenizer_hs.eos_token_id
    id2label_hs = {0: "No", 1: "Yes"}
    model_hs = AutoModelForSequenceClassification.from_pretrained(
        cfg_hs.base_model_name_or_path, return_dict=True, num_labels=len(id2label_hs), id2label=id2label_hs
    )
    model_hs = PeftModel.from_pretrained(model_hs, peft_hs)

    # Sentiment Analysis Model
    peft_sa = "VRLLab/TurkishBERTweet-Lora-SA"
    cfg_sa = PeftConfig.from_pretrained(peft_sa)
    tokenizer_sa = AutoTokenizer.from_pretrained(cfg_sa.base_model_name_or_path, padding_side="right")
    if tokenizer_sa.pad_token_id is None:
        tokenizer_sa.pad_token_id = tokenizer_sa.eos_token_id
    id2label_sa = {0: "negative", 1: "neutral", 2: "positive"}
    model_sa = AutoModelForSequenceClassification.from_pretrained(
        cfg_sa.base_model_name_or_path, return_dict=True, num_labels=len(id2label_sa), id2label=id2label_sa
    )
    model_sa = PeftModel.from_pretrained(model_sa, peft_sa)

    return model_hs, tokenizer_hs, id2label_hs, model_sa, tokenizer_sa, id2label_sa

model_hs, tokenizer_hs, id2label_hs, model_sa, tokenizer_sa, id2label_sa = load_models()

uploaded_docx = st.file_uploader("📄 DOCX dosyanızı yükleyin", type=["docx"])
manual_text = st.text_area("✏️ Veya doğrudan metin yapıştırın", height=200)

sentences = []
if uploaded_docx is not None:
    document = Document(uploaded_docx)
    full_text = "\n".join([para.text for para in document.paragraphs if para.text.strip()])
    sentences = split_sentences(full_text)
elif manual_text.strip():
    sentences = split_sentences(manual_text.strip())

if sentences:
    preprocessed_sentences = [preprocess(s) for s in sentences]
    results = []
    hate_count = 0

    progress = st.progress(0, text="Analiz ediliyor...")
    total = len(sentences)

    with torch.no_grad():
        for i, (sent, preproc) in enumerate(zip(sentences, preprocessed_sentences)):
            ids_hs = tokenizer_hs.encode_plus(preproc, return_tensors="pt", truncation=True)
            label_hs = model_hs(**ids_hs).logits.argmax(-1).item()
            hate = id2label_hs[label_hs]
            if hate == "Yes":
                hate_count += 1

            ids_sa = tokenizer_sa.encode_plus(preproc, return_tensors="pt", truncation=True)
            label_sa = model_sa(**ids_sa).logits.argmax(-1).item()
            sentiment = id2label_sa[label_sa]

            results.append({"sentence": sent, "hate_speech": hate, "sentiment": sentiment})
            progress.progress((i + 1) / total, text=f"İşleniyor ({i+1}/{total})")

    st.success("✅ Analiz tamamlandı!")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    ratio = hate_count / len(results) * 100
    st.metric(label="Nefret Söylemi Oranı", value=f"%{ratio:.2f}", delta=f"{hate_count} / {len(results)} cümle")

    # Duygu grafiği
    sentiment_counts = df['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'], fill_value=0)
    st.subheader("📈 Duygu Dağılımı")
    fig, ax = plt.subplots()
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red'])
    ax.set_ylabel("Cümle Sayısı")
    ax.set_title("Cümlelerin Duygu Etiketlerine Göre Dağılımı")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    st.pyplot(fig)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 CSV olarak indir",
        data=csv,
        file_name="metin_analizi_sonuclari.csv",
        mime="text/csv",
    )
