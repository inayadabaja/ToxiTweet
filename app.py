import os
import re
import torch
import gradio as gr
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# Chemin absolu vers le dossier où se trouve app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensuite, construis les chemins relatifs à partir de BASE_DIR
BERT_MODEL_NAME = os.path.join(BASE_DIR, "fine_tuned_bert_model")
BERT_TOKENIZER_NAME = os.path.join(BASE_DIR, "fine_tuned_bert_tokenizer")
TFIDF_VECTORIZER_NAME = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
LOGREG_MODEL_NAME = os.path.join(BASE_DIR, "logistic_regression_model.pkl")

# --- Chargement du modèle BERT fine-tuned ---
try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
    model.eval()
    print("BERT model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading BERT model/tokenizer: {e}")
    tokenizer = None
    model = None

# --- Initialisation pipeline zero-shot ---
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["hate speech", "offensive language", "neutral"]
print("Zero-shot classifier initialized.")

# --- Chargement TF-IDF + Logistic Regression ---
try:
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_NAME)
    logistic_regression_model = joblib.load(LOGREG_MODEL_NAME)
    print("TF-IDF vectorizer and Logistic Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading TF-IDF/Logistic Regression models: {e}")
    tfidf_vectorizer = None
    logistic_regression_model = None

# --- Nettoyage de texte ---
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # enlever URLs
    text = re.sub(r'@\w+', '', text)     # enlever mentions
    text = re.sub(r'#\w+', '', text)     # enlever hashtags
    text = re.sub(r'[^\w\s]', '', text)  # enlever ponctuation
    text = text.strip()
    return text

# --- Prédiction BERT ---
def predict_bert(text):
    if tokenizer is None or model is None:
        return {"Error": "BERT model not loaded."}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    labels = ["hate speech", "offensive language", "neutral"]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# --- Prédiction zero-shot ---
def predict_zero_shot(text):
    output = zero_shot_classifier(text, candidate_labels)
    return dict(zip(output['labels'], output['scores']))

# --- Prédiction TF-IDF + LogReg ---
def predict_tfidf_logreg(text):
    if tfidf_vectorizer is None or logistic_regression_model is None:
        return {"Error": "TF-IDF model not loaded."}
    cleaned_text = clean_tweet(text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    probs = logistic_regression_model.predict_proba(vectorized_text)[0]
    class_mapping = {
        0: "offensive language",
        1: "hate speech",
        2: "neutral"
    }
    result = {class_mapping[i]: round(probs[i], 3) for i in range(len(probs))}
    return result

# --- Fonction globale de classification ---
def classify(text):
    try:
        bert_output = predict_bert(text)
        zero_shot_output = predict_zero_shot(text)
        tfidf_logreg_output = predict_tfidf_logreg(text)
        return bert_output, zero_shot_output, tfidf_logreg_output
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        return {error_msg}, {error_msg}, {error_msg}

# --- Documentation ---
documentation_markdown = """
# ToxiTweet Demo Documentation

This application demonstrates the classification of tweets into categories: "hate speech", "offensive language", and "neutral" using three different models:
1.  **BERT fine-tuned:** A transformer-based model (bert-base-uncased) fine-tuned on a custom dataset.
2.  **Zero-shot Classification:** A pre-trained model (facebook/bart-large-mnli) that can classify text into categories without explicit training on those categories.
3.  **TF-IDF + Logistic Regression:** A traditional machine learning approach using TF-IDF features and a Logistic Regression classifier as a baseline.

## Data Selection for Training

The models (BERT fine-tuned and TF-IDF + Logistic Regression) were trained on the `labeled_data.csv` dataset. This dataset likely contains tweets labeled with one of the three categories.

## Model Selection & Training Process

### BERT Fine-tuned
* **Base Model:** `bert-base-uncased`
* **Tokenization:** Tweets were tokenized using the `AutoTokenizer` from Hugging Face.
* **Training:** The model was fine-tuned using `Trainer` from the `transformers` library. Key training arguments included a learning rate of `2e-5`, batch size of `16`, and `3` epochs. The best model was loaded based on the F1-score.

### Zero-shot Classification
* **Model:** `facebook/bart-large-mnli`
* **Approach:** This model leverages its understanding of natural language inference to classify text by comparing it against candidate labels. No specific training on this dataset was required for this model.

### TF-IDF + Logistic Regression
* **Feature Extraction:** Tweets were converted into numerical features using `TfidfVectorizer` with a maximum of `5000` features and `(1,2)` n-gram range.
* **Classifier:** A `LogisticRegression` model was trained on the TF-IDF features.

## Models' Performance

The performance of each model was evaluated using standard classification metrics (Accuracy, Precision, Recall, F1-score) and inference time. The results were summarized in a comparison table and visualized.

| Model                        | Accuracy | Precision | Recall | F1-score | Inference time (s) |
|------------------------------|----------|-----------|--------|----------|--------------------|
| TF-IDF + Logistic Regression | 0.8850   | 0.8700    | 0.8850 | 0.8693   | 0.0006             |
| BERT fine-tuned              | 0.9129   | 0.9064    | 0.9129 | 0.9088   | 33.4462            |
| Zero-shot classification     | 0.7529   | 0.7357    | 0.7529 | 0.7363   | 291.4774           |

## How to Use the Demo

Enter a tweet in the text box and click "Submit". The app will display the classification probabilities for each of the three models.
"""

# --- Interface Gradio ---
with gr.Blocks(title="ToxiTweet App") as iface:
    gr.Markdown("## ToxiTweet: Tweet Classification Demo")

    with gr.Tab("Demo"):
        gr.Markdown("Enter a tweet below to classify it using different models.")
        tweet_input = gr.Textbox(lines=3, placeholder="Write your tweet here...", label="Enter Tweet")

        with gr.Row():
            bert_output = gr.Label(num_top_classes=3, label="BERT Fine-tuned Prediction")
            zero_shot_output = gr.Label(num_top_classes=3, label="Zero-shot Prediction")
            tfidf_logreg_output = gr.Label(num_top_classes=3, label="TF-IDF + Logistic Regression Prediction")

        classify_button = gr.Button("Classify Tweet")
        classify_button.click(
            fn=classify,
            inputs=tweet_input,
            outputs=[bert_output, zero_shot_output, tfidf_logreg_output]
        )

    with gr.Tab("Documentation"):
        gr.Markdown(documentation_markdown)

if __name__ == "__main__":
    iface.launch(debug=True)
