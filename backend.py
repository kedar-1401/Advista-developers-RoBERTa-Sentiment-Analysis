import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)
CORS(app)

# Load the tokenizer and model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Define sentiment labels
labels = ['negative', 'neutral', 'positive']

# Function to get sentiment and confidence scores
def get_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = F.softmax(output.logits, dim=1)[0]
    predicted_class = torch.argmax(scores).item()
    return labels[predicted_class], scores.tolist()

# Endpoint for predicting sentiment from single comment
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    sentiment, confidence = get_sentiment(text)
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

# Endpoint for processing comments from a CSV file
@app.route("/upload", methods=["POST"])
def upload_csv():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)

    if "Comment" not in df.columns:
        return jsonify({"error": "CSV must have a 'Comment' column"}), 400

    results = []
    for comment in df["Comment"].fillna(""):
        sentiment, _ = get_sentiment(comment)
        print(sentiment)
        results.append({
            "Comment": comment,
            "Sentiment": sentiment
        })

    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)