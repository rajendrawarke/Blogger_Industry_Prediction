from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

# Load model and embedding generator
model = joblib.load("models/rf_blog_topic_model.pkl")
embedder = SentenceTransformer("models/miniLM_embedding_model")

# Flask setup
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    blog_text = request.form["blogtext"]
    embedding = embedder.encode([blog_text])
    prediction = model.predict(embedding)[0]
    predict_proba = np.max(model.predict_proba(embedding))
    return jsonify({"topic": prediction, "proba":f"{predict_proba:.2%}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
