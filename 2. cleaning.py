import re
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Dataset/IMDB dataset.csv")

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = text.lower()
    # Merge 'not X' into 'not_X'
    text = re.sub(r"\bnot (\w+)", r"not_\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["review"] = df["review"].apply(clean_text)

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

df.to_csv("Dataset/IMDB_cleaned.csv", index=False)