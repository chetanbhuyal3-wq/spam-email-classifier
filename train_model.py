"""
train_model.py
==============
Spam Email Classifier — Training Pipeline
Handles: loading data → preprocessing → TF-IDF → model training → evaluation → save
"""

import os
import re
import pickle
import string
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline

# ── Optional: NLTK stopwords (fallback to manual list if unavailable) ──
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {
        'i','me','my','myself','we','our','ours','ourselves','you','your',
        'yours','yourself','he','him','his','himself','she','her','hers',
        'herself','it','its','itself','they','them','their','theirs',
        'what','which','who','whom','this','that','these','those','am',
        'is','are','was','were','be','been','being','have','has','had',
        'having','do','does','did','doing','a','an','the','and','but',
        'if','or','because','as','until','while','of','at','by','for',
        'with','about','against','between','into','through','during',
        'before','after','above','below','to','from','up','down','in',
        'out','on','off','over','under','again','further','then','once',
        'here','there','when','where','why','how','all','both','each',
        'few','more','most','other','some','such','no','nor','not',
        'only','own','same','so','than','too','very','s','t','can',
        'will','just','don','should','now','d','ll','m','o','re','ve',
        'y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
        'isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
        'weren','won','wouldn'
    }


# ─────────────────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Replace URLs with token
      3. Replace email addresses with token
      4. Replace phone numbers with token
      5. Replace numbers/currency with token
      6. Remove punctuation
      7. Remove stopwords
      8. Strip extra whitespace
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Replace URLs
    text = re.sub(r'http\S+|www\.\S+|https\S+', ' urltoken ', text)

    # Step 3: Replace email addresses
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)

    # Step 4: Replace phone numbers
    text = re.sub(r'[\+\(]?[1-9][0-9\-\(\)\s]{7,}[0-9]', ' phonetoken ', text)

    # Step 5: Replace currency and numbers
    text = re.sub(r'\$[\d,]+', ' moneytoken ', text)
    text = re.sub(r'\d+', ' numtoken ', text)

    # Step 6: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 7: Tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────
# 2. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df.dropna(subset=['label', 'text'], inplace=True)
    df['label'] = df['label'].str.strip().str.lower()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df.dropna(subset=['label_num'], inplace=True)
    df['label_num'] = df['label_num'].astype(int)
    df['clean_text'] = df['text'].apply(preprocess_text)

    print(f"\n{'─'*55}")
    print(f"  📊  Dataset Summary")
    print(f"{'─'*55}")
    print(f"  Total samples   : {len(df)}")
    counts = df['label'].value_counts()
    print(f"  Ham (Safe)      : {counts.get('ham', 0)}")
    print(f"  Spam            : {counts.get('spam', 0)}")
    print(f"{'─'*55}\n")
    return df


# ─────────────────────────────────────────────────────────
# 3. TRAIN MODEL
# ─────────────────────────────────────────────────────────

def train(
    data_path: str  = 'data/emails.csv',
    model_path: str = 'model/spam_model.pkl',
    use_logistic: bool = False
):
    df = load_data(data_path)
    X  = df['clean_text']
    y  = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Build sklearn Pipeline ──────────────────────────
    if use_logistic:
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf_name   = "Logistic Regression"
    else:
        classifier = MultinomialNB(alpha=0.1)
        clf_name   = "Multinomial Naive Bayes"

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams
            max_features=15_000,
            sublinear_tf=True,        # apply 1 + log(tf)
            min_df=1,
            stop_words=None           # already preprocessed
        )),
        ('clf', classifier)
    ])

    pipeline.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc       = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else None
    cm        = confusion_matrix(y_test, y_pred)

    print(f"{'═'*55}")
    print(f"  🤖  Model: {clf_name}")
    print(f"{'═'*55}")
    print(f"  Accuracy       : {acc*100:.2f}%")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1-Score       : {f1:.4f}")
    if roc:
        print(f"  ROC-AUC        : {roc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  ┌─────────────────────────┐")
    print(f"  │ TN={cm[0,0]:>4}   FP={cm[0,1]:>4}   │")
    print(f"  │ FN={cm[1,0]:>4}   TP={cm[1,1]:>4}   │")
    print(f"  └─────────────────────────┘")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print(f"{'═'*55}\n")

    # ── Save ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    meta = {
        'pipeline':   pipeline,
        'metrics': {
            'accuracy': acc, 'precision': precision,
            'recall': recall, 'f1': f1,
            'roc_auc': roc, 'confusion_matrix': cm.tolist()
        },
        'clf_name': clf_name
    }
    with open(model_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"  ✅  Model saved → {model_path}\n")
    return meta


# ─────────────────────────────────────────────────────────
# 4. PREDICT
# ─────────────────────────────────────────────────────────

def predict(text: str, model_path: str = 'model/spam_model.pkl') -> dict:
    """Classify a single email. Returns full result dict."""
    with open(model_path, 'rb') as f:
        meta = pickle.load(f)

    pipeline   = meta['pipeline']
    clean      = preprocess_text(text)
    pred       = pipeline.predict([clean])[0]
    proba      = pipeline.predict_proba([clean])[0]
    spam_score = float(proba[1])
    ham_score  = float(proba[0])

    # Risk level
    if spam_score >= 0.80:
        risk = 'HIGH'
    elif spam_score >= 0.50:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'

    return {
        'label':       'spam' if pred == 1 else 'ham',
        'is_spam':     bool(pred == 1),
        'spam_prob':   round(spam_score * 100, 1),
        'ham_prob':    round(ham_score  * 100, 1),
        'confidence':  round(max(spam_score, ham_score) * 100, 1),
        'risk_level':  risk,
        'clean_text':  clean,
    }


if __name__ == '__main__':
    train()
