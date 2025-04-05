import pandas as pd
import pickle
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
X = scipy.sparse.load_npz("X_features.npz")
y = pd.read_csv("y_labels.csv").values.ravel()  # Flatten to 1D array if needed

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store model accuracies
model_accuracies = {}

# Train and evaluate Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
model_accuracies["Logistic Regression"] = accuracy_score(y_test, log_pred)

# Train and evaluate Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
model_accuracies["Naive Bayes"] = accuracy_score(y_test, nb_pred)

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
model_accuracies["Random Forest"] = accuracy_score(y_test, rf_pred)

# Print model accuracy comparison
print("\n🔍 Model Comparison Results:")
for name, acc in model_accuracies.items():
    print(f"{name}: {acc * 100:.2f}%")

# Determine best model based on accuracy
best_model_name = max(model_accuracies, key=model_accuracies.get)
print(f"\n✅ Best Model: {best_model_name} with {model_accuracies[best_model_name] * 100:.2f}% accuracy!")

# Assign actual model object to best_model
if best_model_name == "Logistic Regression":
    best_model = log_model
elif best_model_name == "Naive Bayes":
    best_model = nb_model
else:
    best_model = rf_model

# Save model using pickle
with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save model using joblib
joblib.dump(best_model, "random_forest_model.pkl")
print("💾 Model saved as both best_model.pkl and random_forest_model.pkl")


