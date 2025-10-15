from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

df = pd.read_csv("Dataset/IMDB_cleaned.csv")


X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

print("✅ Model training complete!")
joblib.dump(model, "sentiment_model.pkl")
print("✅ Model saved as 'sentiment_model.pkl'")