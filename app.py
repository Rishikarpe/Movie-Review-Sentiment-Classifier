import streamlit as st
import joblib

# Load model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("Movie Review Sentiment Classifier")
st.write("Enter a movie review and see if it is positive or negative!")

# Text input
user_input = st.text_area("Your Review:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        # Transform input
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        st.success(f"Predicted Sentiment: {prediction.capitalize()}")
    else:
        st.warning("Please enter a review to predict.")
