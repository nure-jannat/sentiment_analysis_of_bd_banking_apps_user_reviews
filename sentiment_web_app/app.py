from flask import Flask, render_template, request
import joblib
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("combined_tfidf_vectorizer.pkl")

# Stopwords
custom_english_stopwords = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", "o"
]
custom_bengali_stopwords = [
    "আমি", "আমরা", "তুমি", "তোমরা", "সে", "তারা", "এটা", "ওটা", "এই", "সেই", "কে", 
    "কাকে", "কোথায়", "কখন", "কবে", "যখন", "তখন", "যদি", "যদিও", "আর", "এবং", "অথবা", 
    "কিন্তু", "যেন", "যেমন", "তাই", "তো", "হ্যাঁ", "ছিল", "ছিলাম", "হয়", "হবে", 
    "কর", "করেছে", "করা", "করি", "করছেন", "করার", "করা হয়", "করছেন", "করো", "করলাম", 
    "একটি", "এক", "কিছু", "অনেক", "সব", "শুধু", "তবে", "আরও", "যিনি", "যার", "যারাও", 
    "হলো", "হয়ে", "তাকে", "তাদের", "এটি", "ওটি", "ইত্যাদি", "তারপর", "উপর", "নিচে", 
    "থেকে", "জন্য", "সঙ্গে", "মধ্যে", "দিকে", "দুবাই", "লক্ষ্য", "এইটা", "কোন"
]
all_stopwords = set(word.lower() for word in (custom_english_stopwords + custom_bengali_stopwords))

lemmatizer = WordNetLemmatizer()

# Preprocessing functions
def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    ends_with_period = text.endswith('.')
    text = re.sub(r'[.,|।]', ' ', text)
    if ends_with_period:
        text = text.rstrip() + '.'
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_mixed(text):
    return word_tokenize(text)

def remove_stopwords_from_tokens(tokens):
    return [token for token in tokens if token.lower() not in all_stopwords]

def is_english(tokens):
    english_chars = sum(1 for token in tokens for char in token if 'a' <= char.lower() <= 'z')
    bangla_chars = sum(1 for token in tokens for char in token if '\u0980' <= char <= '\u09FF')
    return english_chars >= bangla_chars

def lemmatize_if_english(tokens):
    if is_english(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]
    else:
        return tokens

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review_text = ""
    if request.method == "POST":
        review_text = request.form["review"]
        if review_text.strip() != "":
            cleaned = clean_text(review_text)
            tokens = tokenize_mixed(cleaned)
            filtered = remove_stopwords_from_tokens(tokens)
            lemmatized = lemmatize_if_english(filtered)
            final_text = ' '.join(lemmatized)
            tfidf = vectorizer.transform([final_text])
            pred = model.predict(tfidf)[0]
            prediction = "😊 Positive" if pred == 1 else "☹️ Negative"
    return render_template("index.html", prediction=prediction, review_text=review_text)

if __name__ == "__main__":
    app.run(debug=True)
