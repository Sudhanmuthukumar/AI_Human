from flask import Flask, render_template, request, jsonify
import joblib, re, nltk, numpy as np, textstat
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xgboost as xgb

app = Flask(__name__)

# -----------------------
# Load models
# -----------------------
tfidf_vector = joblib.load("tfidf.pkl")
svd = joblib.load("svd.pkl")
booster = joblib.load("xgb_booster.pkl")
log_reg = joblib.load("log_reg.pkl")

# -----------------------
# Preprocessing setup
# -----------------------
REPLACE_BAD_WORD = re.compile(r'[/(){}\[\]\|@,;]')
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = REPLACE_BAD_WORD.sub(" ", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)

def extract_linguistic_features(text):
    readability = textstat.flesch_reading_ease(text)
    words = text.split()
    vocab_div = len(set(words)) / len(words) if words else 0
    avg_sent_len = np.mean([len(s.split()) for s in text.split('.') if s.strip()]) if '.' in text else len(words)
    return np.array([readability, vocab_div, avg_sent_len])

# Sentiment for display only
def sentiment_score_from_text(text):
   # Extended positive keywords
    pos_keywords = [
    "love", "happy", "good", "great", "amazing", "wonderful", "like", "enjoy",
    "excellent", "cool", "fantastic", "awesome", "brilliant", "delightful", "fabulous",
    "glad", "joyful", "pleased", "cheerful", "optimistic", "thrilled", "blessed",
    "content", "ecstatic", "elated", "energetic", "grateful", "hopeful", "inspired",
    "jubilant", "laugh", "laughing", "smile", "smiling", "satisfied", "stunning",
    "terrific", "thankful", "amused", "blissful", "confident", "peaceful", "proud",
    "radiant", "relieved", "sparkling", "splendid", "vibrant"
]

# Extended negative keywords
    neg_keywords = [
    "bad", "hate", "terrible", "worst", "awful", "sad", "angry", "boring",
    "disgusting", "fear", "annoyed", "disappointed", "frustrated", "guilty",
    "jealous", "lonely", "mad", "nervous", "pain", "regret", "resentful",
    "stressed", "upset", "unhappy", "anxious", "depressed", "embarassed",
    "hurt", "miserable", "pessimistic", "resentment", "shocked", "sorrow",
    "tense", "trouble", "upset", "vulnerable", "worried", "tired", "dreadful",
    "hopeless", "offended", "uncomfortable", "grief", "discouraged", "frightened"
]

    text_lower = text.lower()
    pos = sum(word in text_lower for word in pos_keywords)
    neg = sum(word in text_lower for word in neg_keywords)
    return pos - neg

def predict_text(text):
    cleaned = clean_text(text)
    tfidf_vec = tfidf_vector.transform([cleaned])
    svd_features = svd.transform(tfidf_vec)
    ling = extract_linguistic_features(text).reshape(1, -1)
    final_features = np.hstack([svd_features, ling])

    dmatrix = xgb.DMatrix(final_features)
    xgb_proba = booster.predict(dmatrix)
    lr_proba = log_reg.predict_proba(final_features)[:, 1]
    avg_proba = (xgb_proba + lr_proba) / 2

    ai_score = float(avg_proba[0])
    human_score = 1.0 - ai_score

    if 0.4 <= ai_score <= 0.6:
        label = "Uncertain"
    elif ai_score > 0.6:
        label = "AI"
    else:
        label = "Human"

    sentiment_val = sentiment_score_from_text(text)
    sentiment = "Positive" if sentiment_val > 0 else "Negative" if sentiment_val < 0 else "Neutral"
    readability, vocab_div, avg_sent_len = ling[0]

    # -----------------------
    # Round off numeric values
    # -----------------------
    return {
        "label": label,
        "ai_score": round(ai_score*100, 2),
        "human_score": round(human_score*100, 2),
        "readability": round(float(readability), 2),
        "vocab_diversity": round(float(vocab_div), 2),
        "avg_sentence_length": round(float(avg_sent_len), 2),
        "sentiment": sentiment
    }

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    result = predict_text(text)
    return jsonify(result)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
