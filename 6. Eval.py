import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 2️⃣ Load your dataset (can be full or just test portion)
df = pd.read_csv("Dataset/IMDB_cleaned.csv")  # make sure this is your cleaned dataset

# 3️⃣ Split into train/test the same way as before
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

# 4️⃣ Transform test data using TF-IDF
X_test_tfidf = vectorizer.transform(X_test)

# 5️⃣ Predict using the loaded model
y_pred = model.predict(X_test_tfidf)

# 6️⃣ Evaluate
print("✅ Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
