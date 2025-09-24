from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect
from googletrans import Translator
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib, requests, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
translator = Translator()
keyword_model = KeyBERT()

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_news(news: NewsInput):
    raw_text = news.text
    lang = detect(raw_text)

    if lang != 'en':
        translated_obj = await translator.translate(raw_text, dest='en')
        translated = translated_obj.text
    else:
        translated = raw_text

    X = vectorizer.transform([translated])
    y_pred = model.predict(X)[0]
    y_prob = model.predict_proba(X)[0][y_pred]

    keywords = keyword_model.extract_keywords(translated, top_n=3)
    top_kw = [k[0] for k in keywords if isinstance(k, tuple) and len(k) > 1 and k[1] > 0.3]
    related_news = []

    if y_pred == 1 and top_kw:
        query = '+'.join(top_kw)
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={os.getenv('NEWS_API_KEY')}"
        try:
            res = requests.get(url).json()
            for item in res.get("articles", [])[:3]:
                related_news.append({"title": item['title'], "url": item['url']})
        except:
            pass

    return {
        "prediction": int(y_pred),
        "probability": float(y_prob),
        "language": lang,
        "translated_text": translated if lang != 'en' else "",
        "related_news": related_news
    }