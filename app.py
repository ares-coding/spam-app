import streamlit as st
import tensorflow as tf
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = tf.keras.models.load_model("spam_classifier.keras")
tfidf = joblib.load("tfidf_vectorizer.pkl")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c == ' '])
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_spam(text):
    text = clean_text(text)
    vector = tfidf.transform([text]).toarray()
    pred = model.predict(vector)[0][0]
    return pred

st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Spam Message Detector")
st.write("Paste a message below to check if it is **Spam** or **Not Spam (Ham)**.")

user_input = st.text_area(
    "Enter your message:",
    placeholder="e.g. Congratulations! You won ₱50,000. Click the link to claim your prize!"
)


if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        pred = predict_spam(user_input)
        confidence = pred * 100

        if pred > 0.5:
            st.error(f"🚨 This message is SPAM ({confidence:.2f}%)")
        else:
            st.success(f"✅ This message is NOT spam (HAM) ({100 - confidence:.2f}%)")


