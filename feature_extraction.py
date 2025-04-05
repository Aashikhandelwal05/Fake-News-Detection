import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import scipy.sparse

# Load the cleaned dataset
df = pd.read_csv("cleaned_news_dataset.csv")

# Drop any rows with missing values
df.dropna(subset=["text"], inplace=True)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Keep top 5000 words
X = vectorizer.fit_transform(df["text"].astype(str))  # Convert to string to avoid NaN errors

# Save TF-IDF vectorizer for future use
# Save
import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Convert labels to numbers
y = df["label"]  # Assuming label column exists (1 = Fake, 0 = Real)

# Save processed features
scipy.sparse.save_npz("X_features.npz", X)
y.to_csv("y_labels.csv", index=False)

print("✅ Feature extraction complete! TF-IDF features saved.")

