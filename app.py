from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    result = None
    if request.method == "POST":
        input_text = request.form["news_text"]
        if input_text.strip() != "":
            cleaned = clean_text(input_text)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            result = "FAKE" if prediction == 1 else "REAL"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
