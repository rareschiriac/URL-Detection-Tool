import streamlit as st
import joblib
import pandas as pd
from features import extract_features, explain_url, FEATURE_COLS
from urllib.parse import urlparse

st.set_page_config(
    page_title="URL Phishing Detector",
    layout="centered"
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

model = joblib.load("models/url_model.joblib")
feature_cols = joblib.load("models/feature_cols.joblib")

TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "linkedin.com", "amazon.com", "apple.com", "microsoft.com", "github.com",
    "stackoverflow.com", "wikipedia.org", "reddit.com", "netflix.com", "bbc.co.uk",
    "nhs.uk", "gov.uk", "yahoo.com", "whatsapp.com", "tiktok.com",
    "ebay.com", "paypal.com", "dropbox.com", "adobe.com", "spotify.com",
    "twitch.tv", "pinterest.com", "tumblr.com", "wordpress.com", "blogger.com"
}

def get_root_domain(url: str) -> str:
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").lower()
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    except Exception:
        return ""

if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">🛡️</div>
        <div class="welcome-title">URL <span>Phishing</span> Detector</div>
        <div class="welcome-divider"></div>
        <div class="welcome-sub">
            A machine learning tool that analyses URLs in real time and determines
            whether they are safe or potentially malicious. Built to detect phishing
            attacks before they cause harm.
        </div>
        <div class="welcome-stats">
            <div class="stat-item">
                <div class="stat-number">549K+</div>
                <div class="stat-label">URLs Trained On</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">92.7%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">26</div>
                <div class="stat-label">Features Analysed</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started"):
            st.session_state.started = True
            st.rerun()

else:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1>URL Phishing Detector</h1>
        <p>Analyse any URL instantly to determine whether it is safe or malicious</p>
        <span class="tag">Powered by Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    url = st.text_input("", placeholder="Enter a URL to analyse — e.g. https://example.com")
    analyse = st.button("Analyse URL")
    st.markdown('</div>', unsafe_allow_html=True)

    if analyse:
        if not url.strip():
            st.warning("Please enter a URL before analysing.")
        else:
            root_domain = get_root_domain(url)
            trusted = root_domain in TRUSTED_DOMAINS

            if trusted:
                pred = 0
            else:
                feats = extract_features(url)
                X = pd.DataFrame([[feats.get(c, 0) for c in feature_cols]], columns=feature_cols)
                proba = model.predict_proba(X)[0][1]
                pred = 1 if proba >= 0.60 else 0

            if pred == 1:
                st.markdown("""
                <div class="result-box result-malicious">
                    <div class="result-title">MALICIOUS / PHISHING</div>
                    <div class="result-description">
                        This URL has been classified as malicious. The model identified a combination
                        of suspicious characteristics in the structure of this URL that are commonly
                        associated with phishing attacks. It is recommended that you do not visit
                        this site or enter any personal information.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box result-safe">
                    <div class="result-title">SAFE</div>
                    <div class="result-description">
                        This URL has been classified as safe. The structure and characteristics
                        of this URL are consistent with legitimate websites. No significant
                        phishing indicators were detected.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            reasons = explain_url(url)
            reasons_html = "".join(f'<div class="reason-item">{r}</div>' for r in reasons)
            st.markdown(f"""
            <div class="reasons-box">
                <h3>Analysis Breakdown</h3>
                {reasons_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Show extracted features"):
                feats = extract_features(url)
                feat_df = pd.DataFrame({
                    "Feature": list(feats.keys()),
                    "Value": list(feats.values())
                })
                st.dataframe(feat_df, use_container_width=True)

    st.markdown("""
    <div class="footer">
        URL Phishing Detector — Final Year Project
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
