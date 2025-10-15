import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Try predicting on new text
sample = input("Enter your review: ")
sample_tfidf = vectorizer.transform([sample])
prediction = model.predict(sample_tfidf)

print("Predicted Sentiment:", prediction[0])
