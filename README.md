# Movie-Review-Sentiment-Classifier
A simple **Sentiment Analysis** project using **TF-IDF + Logistic Regression** to classify movie reviews as **Positive** or **Negative**. This project also demonstrates model saving/loading, evaluation, and visualization using a confusion matrix.

---

## Project Overview
This project trains a sentiment analysis model on the **IMDB movie reviews dataset**.  
It starts with **TF-IDF vectorization** and a **Logistic Regression classifier**, which serves as a baseline.  
The trained model can predict the sentiment of new movie reviews.

---
## Dataset
- **Source:** [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Size:** 50,000 reviews (25,000 positive + 25,000 negative)  
- **Format:** CSV with columns:
  - `review` — text of the review  
  - `sentiment` — label (`positive` / `negative`)

---
## Features
- Text cleaning (remove HTML tags, punctuation, lowercase)  
- TF-IDF vectorization  
- Logistic Regression classifier  
- Model saving/loading with `joblib`  
- Evaluation using **accuracy**, **classification report**, and **confusion matrix**

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/sentiment-classifier.git
cd sentiment-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

1️⃣ Train & Save Model
```bash
python train_model.py
```

```bash
2️⃣ Evaluate Model
python eval_model.py
```
```bash
3️⃣ Predict Sentiment for New Reviews
python test.py
# Example input:
# Enter your review: The movie was fantastic!
# Predicted Sentiment: positive
```

---

## Evaluation

 - Accuracy: ~89.8%
 - Classification report:
 - Positive reviews: Precision ~0.89, Recall ~0.92
 - Negative reviews: Precision ~0.91, Recall ~0.88
 - Confusion matrix visualizes True Positives: , True Negatives, False Positives, False Negatives.
<img src=https://github.com/Rishikarpe/Movie-Review-Sentiment-Classifier/blob/main/Figure_1.png>
  
---
### Results

- The model performs well on standard positive/negative reviews.
- Fails in some negation cases (e.g., "not good" → predicted positive).
- Ideal next step: upgrade to transformers (e.g., DistilBERT) for better contextual understanding.
---

### Future Improvements
- Use BERT/DistilBERT for better handling of negations and complex sentences
- Handle emoji and slang in reviews
- Deploy as a web app or REST API for real-time predictions

