# 📧 Spam Email Classifier

A full-stack Machine Learning web application that detects whether an email is **Spam ⚠** or **Safe ✅** using NLP and Naive Bayes.

---

## 📁 Folder Structure

```
SpamEmailClassifier/
│
├── app.py                      ← Flask web server (routes + API)
├── train_model.py              ← ML pipeline: preprocess → TF-IDF → train → evaluate → save
├── requirements.txt            ← Python dependencies
│
├── data/
│   └── emails.csv              ← Training dataset (label, text)
│
├── model/
│   └── spam_model.pkl          ← Saved sklearn Pipeline (auto-generated)
│
├── templates/
│   └── index.html              ← Full frontend: HTML + CSS + JavaScript
│
├── architecture_diagram.svg    ← System architecture diagram
├── block_diagram.svg           ← ML block flow diagram
└── README.md                   ← This file
```

---

## 🗃️ Dataset Format

The training dataset `data/emails.csv` uses two columns:

| Column | Type   | Values         | Description           |
|--------|--------|----------------|-----------------------|
| label  | string | `ham` / `spam` | Email classification  |
| text   | string | Any text       | Raw email content     |

**Example rows:**
```csv
label,text
ham,"Hi Sarah, the meeting is scheduled for Thursday at 2pm."
spam,"CONGRATULATIONS! You've won $1,000,000! Click here NOW to claim!!!"
ham,"Please review the attached Q3 report and share feedback."
spam,"FREE MEDS no prescription needed. 90% off. Order today!"
```

**Tips for better accuracy:**
- Use 500+ samples (try the UCI SMS Spam Collection dataset)
- Keep ham/spam roughly balanced (50/50 ideal)
- Include diverse examples (phishing, prize scams, adult content, etc.)

---

## 🚀 How to Run

### Step 1 — Navigate to project folder
```bash
cd SpamEmailClassifier
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Activate:
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — (Optional) Train the model and view metrics
```bash
python train_model.py
```
This will print:
- Dataset summary
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Full Classification Report
- Save model to `model/spam_model.pkl`

### Step 5 — Start the Flask server
```bash
python app.py
```
> The model auto-trains on first run if `spam_model.pkl` doesn't exist.

### Step 6 — Open in browser
```
http://localhost:5000
```

---

## 🌐 API Endpoints

| Method | Endpoint    | Description                         |
|--------|-------------|-------------------------------------|
| GET    | `/`         | Serve the web UI                    |
| POST   | `/classify` | Classify email text                 |
| POST   | `/retrain`  | Retrain model with current dataset  |
| GET    | `/metrics`  | Return saved model metrics as JSON  |

### Example `/classify` Request
```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Win a free iPhone! Click here NOW!!!"}'
```

### Example Response
```json
{
  "label": "spam",
  "is_spam": true,
  "spam_prob": 97.3,
  "ham_prob": 2.7,
  "confidence": 97.3,
  "risk_level": "HIGH"
}
```

---

## 🧠 ML Pipeline

```
Raw Email Text
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PREPROCESSING  (train_model.py → preprocess_text())    │
│  • Convert to lowercase                                  │
│  • Replace URLs  →  "urltoken"                          │
│  • Replace email addresses  →  "emailtoken"             │
│  • Replace phone numbers  →  "phonetoken"               │
│  • Replace currency/numbers  →  "moneytoken/numtoken"   │
│  • Remove punctuation                                    │
│  • Remove stopwords (NLTK English)                      │
│  • Tokenize and filter short tokens                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  TF-IDF VECTORIZER  (sklearn TfidfVectorizer)           │
│  • ngram_range = (1, 2)  → unigrams + bigrams           │
│  • max_features = 15,000                                │
│  • sublinear_tf = True  → 1 + log(tf) scaling           │
│  • Output: sparse numeric feature matrix                 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  MULTINOMIAL NAIVE BAYES  (sklearn MultinomialNB)       │
│  • alpha = 0.1  (Laplace smoothing)                     │
│  • Applies Bayes theorem: P(spam | words)               │
│  • Output: class probabilities [ham_prob, spam_prob]    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RESULT                                                  │
│  • label: "spam" or "ham"                               │
│  • spam_prob / ham_prob  (percentage)                   │
│  • risk_level: HIGH / MEDIUM / LOW                      │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluation Metrics

| Metric         | Formula                              | What it means                                |
|----------------|--------------------------------------|----------------------------------------------|
| **Accuracy**   | (TP+TN) / Total                      | % of emails correctly classified             |
| **Precision**  | TP / (TP + FP)                       | Of predicted spam, how many are truly spam?  |
| **Recall**     | TP / (TP + FN)                       | Of all real spam, how many did we catch?     |
| **F1-Score**   | 2 × (P × R) / (P + R)               | Harmonic mean of precision and recall        |
| **Confusion Matrix** | TN / FP / FN / TP grid         | Detailed breakdown of predictions            |

---

## 🔔 Alert System

| Result | Popup Message |
|--------|---------------|
| Spam   | ⚠ Warning: This email is SPAM. Do not open links. |
| Safe   | ✅ Safe Email: This message appears legitimate.   |

Alerts are shown as a full-screen overlay modal with animated entry. Press **Dismiss** or **Escape** to close.

---

## 💡 Improve Accuracy

1. **More data** — add 1000+ samples; try [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
2. **Swap classifier** — replace `MultinomialNB` with `LogisticRegression` or `SVC` in `train_model.py`
3. **Use LogisticRegression** — pass `use_logistic=True` to the `train()` function
4. **Add lemmatization** — install `spacy` and lemmatize tokens in `preprocess_text()`
5. **Expand stopwords** — customize `STOPWORDS` set in `train_model.py`
