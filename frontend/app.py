import streamlit as st
import requests
from utils.auth import check_login
from utils.translator import detect_language
from PIL import Image
import base64
import speech_recognition as sr
import io
import tempfile
import os

API_URL = "http://localhost:8000/predict"  # Update to deployed backend URL

st.set_page_config(page_title="📰 Fake News Detector", layout="wide")

# ---------------- Header ------------------
st.markdown("""
    <style>
        .headline {
            font-size: 2.5em;
            font-weight: bold;
            color: #2E86C1;
        }
        .subtext {
            color: gray;
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="headline">📰 Multilingual Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Check whether a news is fake, from any major language. Built with ML & ❤</div>', unsafe_allow_html=True)

# --------------- Login --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.authenticated = True
            st.success("✅ Logged in!")
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# -------------- News Input ----------------
st.markdown("### 🧾 Input Your News")
input_method = st.radio("Choose input method:", ["Text", "Image", "Voice"], horizontal=True)

news_text = ""

if input_method == "Text":
    news_text = st.text_area("Paste the news article here", height=150)

elif input_method == "Image":
    uploaded_file = st.file_uploader("Upload image of news article", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        import pytesseract
        news_text = pytesseract.image_to_string(image)
        st.text_area("Extracted Text", news_text, height=150)

elif input_method == "Voice":
    uploaded_audio = st.file_uploader("Upload audio (.wav)", type=["wav"])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            tmp_path = tmp.name
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
            try:
                news_text = recognizer.recognize_google(audio)
                st.text_area("Transcribed Text", news_text, height=150)
            except:
                st.warning("❗Could not recognize speech")

# -------------- Predict ----------------
if st.button("🚀 Predict"):
    if not news_text.strip():
        st.warning("Please enter or upload news content.")
        st.stop()

    with st.spinner("Analyzing..."):
        try:
            response = requests.post(API_URL, json={"text": news_text})
            result = response.json()

            label = "✅ Real" if result["prediction"] == 1 else "❌ Fake"
            prob = round(result["probability"] * 100, 2)
            lang = result.get("language", "unknown")
            st.success(f"*Prediction:* {label} ({prob}% confidence)")
            st.info(f"🗣 Language Detected: {lang}")

            if result.get("translated_text"):
                st.markdown("*Translated to English:*")
                st.write(result["translated_text"])

            if result.get("related_news"):
                st.markdown("🔍 Related Real News:")
                for item in result["related_news"]:
                    st.markdown(f"- [{item['title']}]({item['url']})")

            if result["prediction"] == 0:
                st.markdown('<h2 style="color:red">🚨 Warning: This may be Fake News!</h2>', unsafe_allow_html=True)

        except Exception as e:
            st.error("Prediction failed.")
            st.text(str(e))

# ------------- Footer --------------
st.markdown("---")
st.markdown("📘 Made with Streamlit | 🔐 Auth-enabled | 🌐 Multilingual | 🧠 ML-backed")