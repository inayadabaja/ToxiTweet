# ToxiTweet

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

## video demo

🎥 [▶️ Watch Local Deployment Demo](demo/demo_local.mp4)
🎥 [▶️ Watch Local Deployment Demo](demo/demo_huggingFace.mp4)
