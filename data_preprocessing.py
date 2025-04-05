import pandas as pd

# Load datasets (since they are in the main folder)
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Add labels manually
fake_df["label"] = 1  # Fake news
true_df["label"] = 0  # Real news

# Merge both datasets
df = pd.concat([fake_df, true_df])

# Shuffle and reset index
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the updated dataset
df.to_csv("combined_news.csv", index=False)

print(df.head(), df["label"].value_counts())  # Check dataset

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load the dataset
df = pd.read_csv("combined_news.csv")  # Use your latest dataset

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & numbers
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(words)

# Apply cleaning to all text data
df["text"] = df["text"].astype(str).apply(clean_text)

# Save cleaned dataset
df.to_csv("cleaned_news_dataset.csv", index=False)

print("✅ Data cleaning completed! Cleaned file saved as 'cleaned_news_dataset.csv'")


