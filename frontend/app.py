# streamlit_app.py
import streamlit as st
from utils.news_predictor import load_model # Ensure 'predict_news' exists in utils/news_predictor.py
from utils.auth import login_user, register_user
model=load_model("fake_news_model.pkl")
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
# User input
user_input = st.text_area("Enter news text to check:")
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = ""

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
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        name = st.text_input("Name")
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            success, msg = register_user(new_email, new_password, name)
            st.success(msg) if success else st.error(msg)

else:
    st.sidebar.success(f"👤 Welcome, {st.session_state.user_name}")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

    st.title("📰 Fake News Detection")
    headline = st.text_area("📝 Enter News Text", height=100)

    if st.button("Predict"):
      if user_input.strip():
        # Preprocess input (if your model requires vectorizer, add it here)
        prediction = model.predict([user_input])
        result = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
        st.subheader(f"Result: {result}")
      else:
        st.warning("Please enter some text before predicting.")
         
# Predict button
