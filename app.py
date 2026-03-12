"""
app.py
======
Flask Web Application — Spam Email Classifier
Routes: / (UI)  |  POST /classify  |  POST /retrain  |  GET /metrics
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify
from train_model import predict, train

app = Flask(__name__)

MODEL_PATH = 'model/spam_model.pkl'


def ensure_model():
    """Auto-train if no saved model exists."""
    if not os.path.exists(MODEL_PATH):
        print("\n⚙️  No model found — training now...\n")
        train(model_path=MODEL_PATH)


def get_metrics():
    """Return saved evaluation metrics."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            meta = pickle.load(f)
        m = meta.get('metrics', {})
        return {
            'accuracy':  round(m.get('accuracy',  0) * 100, 2),
            'precision': round(m.get('precision', 0) * 100, 2),
            'recall':    round(m.get('recall',    0) * 100, 2),
            'f1':        round(m.get('f1',        0) * 100, 2),
            'clf_name':  meta.get('clf_name', 'Naive Bayes'),
        }
    except Exception:
        return {}


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route('/')
def index():
    metrics = get_metrics()
    return render_template('index.html', metrics=metrics)


@app.route('/classify', methods=['POST'])
def classify():
    data       = request.get_json(silent=True) or {}
    email_text = (data.get('email_text') or '').strip()

    if not email_text:
        return jsonify({'error': 'No email text provided.'}), 400
    if len(email_text) < 5:
        return jsonify({'error': 'Email text too short.'}), 400

    try:
        result = predict(email_text, model_path=MODEL_PATH)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({'error': 'Model not found. Please train first.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        meta = train(model_path=MODEL_PATH)
        m    = meta['metrics']
        return jsonify({
            'message':  f"Model retrained successfully!",
            'accuracy': round(m['accuracy'] * 100, 2),
            'f1':       round(m['f1'] * 100, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    return jsonify(get_metrics())


if __name__ == '__main__':
    ensure_model()
    app.run(debug=True, port=5000)
