import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
import numpy as np

# Load VADER
sia = SentimentIntensityAnalyzer()

# Load RoBERTa
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)

labels = ['negative', 'neutral', 'positive']

st.title("📊 Sentiment Analysis App")
st.write("Analyze text sentiment using both VADER and RoBERTa models.")

text_input = st.text_area("Enter your review/text here:", height=150)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # VADER analysis
        vader_scores = sia.polarity_scores(text_input)
        vader_sentiment = max(vader_scores, key=vader_scores.get)

        # RoBERTa analysis
        encoded_input = tokenizer(text_input, return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        roberta_result = {f"roberta_{label}": round(score, 3) for label, score in zip(labels, scores)}
        roberta_sentiment = labels[np.argmax(scores)]

        # Show results
        st.subheader("VADER Sentiment")
        st.json(vader_scores)
        st.write(f"**VADER Sentiment:** `{vader_sentiment.capitalize()}`")

        st.subheader("RoBERTa Sentiment")
        st.json(roberta_result)
        st.write(f"**RoBERTa Sentiment:** `{roberta_sentiment.capitalize()}`")

        st.subheader("🧠 Final Verdict")
        if vader_sentiment == roberta_sentiment:
            st.success(f"Both agree: **{vader_sentiment.upper()}**")
        else:
            st.info(f"Mixed: VADER → **{vader_sentiment.upper()}**, RoBERTa → **{roberta_sentiment.upper()}**")
