from flask import Flask, request, render_template, jsonify, url_for
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the Flask application
app = Flask(__name__)

# Load the pickled SentimentIntensityAnalyzer model
with open('sia.pkl', 'rb') as file:
    sia = pickle.load(file)

# Define a function to map compound score to emoji
def get_sentiment_emoji(compound_score):
    if compound_score >= 0.05:
        return "😊"  # Positive sentiment
    elif compound_score <= -0.05:
        return "😠"  # Negative sentiment
    else:
        return "😐"  # Neutral sentiment

# Define the route for the home page (Sentiment Analysis)
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.form['text']

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Get the appropriate emoji for the sentiment
    sentiment_emoji = get_sentiment_emoji(sentiment_scores['compound'])

    # Return the results to the template along with the images
    return render_template('index.html', text=text, sentiment=sentiment_scores, emoji=sentiment_emoji)

# Define a route for EDA (Exploratory Data Analysis)
@app.route('/eda')
def eda():
    # List of image filenames for EDA
    images = [
        'day.png', 'dis.png', 'missing_value.png',
        'negative.png', 'neutral.png', 'positive.png',
        'rating_dis.png', 'sentiment.png', 'source_dis.png'
    ]

    return render_template('eda.html', images=images)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
