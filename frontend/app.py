import streamlit as st
import joblib
import os
from utils.auth import login_user, register_user

# --- Paths ---
MODEL_PATH = "C:\\Users\\amrit\\Documents\\GitHub\\NEWS_DETECTION\\models\\fake_news_model.pkl"
VECTORIZER_PATH = "C:\\Users\\amrit\\Documents\\GitHub\\NEWS_DETECTION\\models\\vectorizer.pkl"

# --- Load model ---
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.stop()
else:
    model = joblib.load(MODEL_PATH)

# --- Load vectorizer if needed ---
vectorizer = None
if not hasattr(model, "predict") or "pipeline" not in str(type(model)).lower():
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        st.error("❌ Vectorizer not found. Cannot process text without it.")
        st.stop()

# --- Streamlit config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# --- Session state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = ""

# --- User input ---
user_input = st.text_area("📝 Enter News Text", height=100)

# --- LOGIN / REGISTER ---
if not st.session_state.logged_in:
    st.title("🔐 Login to Fake News Detector")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, msg = login_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_name = msg
            else:
                st.error(msg)

    with tab2:
        name = st.text_input("Name")
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            success, msg = register_user(new_email, new_password, name)
            st.success(msg) if success else st.error(msg)

# --- LOGGED-IN USER INTERFACE ---
else:
    st.sidebar.success(f"👤 Welcome, {st.session_state.user_name}")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

    st.title("📰 Fake News Detection")

    if st.button("Predict"):
        if user_input.strip():
            try:
                # Pipeline prediction
                X_input = [user_input]
                prediction = model.predict(X_input)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
            except ValueError:
                # Separate vectorizer + classifier
                if vectorizer:
                    X_input = vectorizer.transform([user_input])
                    prediction = model.predict(X_input)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_input)[0]
                else:
                    st.error("Cannot predict: No vectorizer available.")
                    st.stop()

            # --- Display results ---
            result = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
            st.subheader(f"Result: {result}")

            # Display confidence if available
            if "proba" in locals():
                st.write(f"Confidence Scores:")
                st.write(f"- Real News: {proba[1]*100:.2f}%")
                st.write(f"- Fake News: {proba[0]*100:.2f}%")
        else:
            st.warning("Please enter some text before predicting.")
