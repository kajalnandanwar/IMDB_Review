
from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (if not already downloaded in model_building.py)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Load the vectorizer and the best model (Logistic Regression in this case)
vectorizer = joblib.load('count_vectorizer.pkl')
model = joblib.load('logistic_regression_model.pkl')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function (same as in model_building.py)
def preprocess_text_for_prediction(text):
    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Remove numbers
    number_pattern = re.compile(r'\\d+')
    text = number_pattern.sub(r'', text)

    # Remove extra spaces
    space_pattern = re.compile(r'\\s+')
    text = space_pattern.sub(r' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)

    # Lemmatization
    tokens = text.split()
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(lemmas)

    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data['review']

    # Preprocess the input review
    cleaned_review = preprocess_text_for_prediction(review)

    # Vectorize the cleaned review
    review_vector = vectorizer.transform([cleaned_review])

    # Make prediction
    prediction = model.predict(review_vector)[0]
    prediction_proba = model.predict_proba(review_vector).tolist()

    return jsonify({
        'sentiment': prediction,
        'probabilities': prediction_proba
    })

@app.route('/')
def home():
    return "Sentiment Analysis API. Send a POST request to /predict with a 'review' in JSON format."

if __name__ == '__main__':
    # In a production environment, use a production-ready WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)
