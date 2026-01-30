import streamlit as st
import joblib
import pandas as pd

from features import explain_url, extract_dataset_features

st.set_page_config(page_title="URL Phishing Detector", page_icon="🔎")

st.title("🔎 URL Phishing Detector")
st.write("Paste a URL to check whether it is **Safe** or **Phishing / Malicious**.")

model = joblib.load("models/url_model.joblib")

FEATURE_COLS = [
    "url_length", "valid_url", "at_symbol", "sensitive_words_count", "path_length",
    "isHttps", "nb_dots", "nb_hyphens", "nb_and", "nb_or", "nb_www", "nb_com", "nb_underscore"
]

url = st.text_input("Enter URL", placeholder="https://example.com/login")

if st.button("Analyze"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        feats = extract_dataset_features(url)
        X = pd.DataFrame([[feats[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]  # probability of malicious (class 1)

        if pred == 1:
            st.error(f"Class: **MALICIOUS / PHISHING**\n\nConfidence (malicious): **{proba:.2%}**")
        else:
            st.success(f"Class: **SAFE**\n\nConfidence (safe): **{(1 - proba):.2%}**")

        st.subheader("Why this result?")
        for reason in explain_url(url):
            st.write(f"- {reason}")

        with st.expander("Show extracted features"):
            st.write(feats)
