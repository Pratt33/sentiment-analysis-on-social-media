import streamlit as st
import joblib

model = joblib.load("sentiment-analysis/output/best_model.pkl")
vectorizer = joblib.load("sentiment-analysis/output/vectorizer.pkl")

st.title("Sentiment Analysis App")
text = st.text_area("Enter text:")
if st.button("Predict"):
    if text.strip():
        vect_text = vectorizer.transform([text])
        pred = model.predict(vect_text)
        sentiment = "Positive" if pred[0] == 1 else "Negative"
        st.write("Sentiment:", sentiment)
    else:
        st.warning("Please enter some text.") 