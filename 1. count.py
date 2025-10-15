import pandas as pd

# Path to your dataset (update this if needed)
df = pd.read_csv("Dataset/IMDB dataset.csv")

print(df.head())
print(df['sentiment'].value_counts())
