# Fake News Style-Risk Detection

A production-grade Machine Learning pipeline for **Fake News Style-Risk Detection**, designed with a strong focus on **honest validation, dataset leakage prevention, reproducibility, and MLOps best practices**.

Rather than treating fake news detection as a simple binary classification problem, this project focuses on identifying **stylistic risk signals** while explicitly controlling for dataset artifacts and source leakage.

---

## 🚀 Key Features

* **Leakage-Controlled Text Preprocessing**

  * Removes URLs, emails, and non-ASCII characters.
  * Removes publisher/source artifacts such as Reuters datelines.
  * Prevents the model from simply memorizing source-specific patterns.

* **End-to-End ML Pipeline**

  * `TextCleaner` → `TF-IDF` → `Logistic Regression`
  * Uses sublinear TF scaling and n-grams.
  * Uses balanced class weights to handle class imbalance.

* **Robust Model Validation**

  * Stratified K-Fold Cross-Validation.
  * Separate holdout test evaluation.
  * Out-of-source evaluation using `GroupShuffleSplit`.

* **Source Confounding Analysis**

  * Evaluates whether metadata such as `subject` or publisher is strongly correlated with the target label.
  * Tests whether the model generalizes beyond the sources present during training.

* **Model Integrity & Compatibility**

  * Generates SHA-256 checksums for trained model artifacts.
  * Detects corrupted or modified model files.
  * Handles Scikit-Learn version compatibility during model loading.

* **Uncertainty-Aware Predictions**

  * Uses configurable decision thresholds.
  * Predictions close to the decision boundary can be classified as `UNCERTAIN` instead of forcing a binary decision.

* **CLI & Interactive Web Application**

  * Command-line inference through `detect_fake_news.py`.
  * Interactive Streamlit dashboard through `app.py`.

---

## 🧠 How It Works

The project follows the pipeline:

```text
Raw News Dataset
      │
      ▼
Data Cleaning & Deduplication
      │
      ▼
Leakage Analysis
(Reuters / Source Artifacts)
      │
      ▼
TextCleaner
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression
      │
      ├───────────────┐
      ▼               ▼
Cross Validation   Holdout Test
                      │
                      ▼
             Out-of-Source Testing
                      │
                      ▼
             Evaluation & Diagnostics
                      │
                      ▼
          ┌───────────┴───────────┐
          ▼                       ▼
      CLI Inference          Streamlit App
```

---

## 📊 Dataset

This project is designed to work with the **ISOT Fake News Dataset**, containing separate datasets for real and fake news articles.

Expected dataset structure:

```text
data/
├── True.csv
└── Fake.csv
```

The preprocessing pipeline performs:

1. Loading real and fake articles.
2. Combining relevant text fields.
3. Removing duplicate articles.
4. Cleaning textual content.
5. Removing known source artifacts.
6. Preparing data for model training.

### ⚠️ Dataset Leakage

A major challenge with the ISOT dataset is **source leakage**.

For example, many real articles contain a Reuters-style dateline such as:

```text
WASHINGTON (Reuters) -
```

If this information is left in the training data, a model can achieve artificially high performance simply by learning that:

```text
Reuters → Real
```

rather than learning meaningful writing-style characteristics.

The `--strip-source-artifacts` option removes these patterns before training.

---

## 🛠️ Tech Stack

* **Python**
* **Scikit-Learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Joblib**
* **Streamlit**

### Machine Learning

* TF-IDF Vectorization
* Logistic Regression
* Stratified K-Fold Cross-Validation
* GroupShuffleSplit
* ROC-AUC
* Precision-Recall Analysis
* Confusion Matrix

---

## 📁 Project Structure

```text
.
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── charts/
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── confusion_matrix.png
│
├── text_clean.py
├── train_model.py
├── evaluation.py
├── model_compat.py
├── detect_fake_news.py
├── app.py
│
├── pipeline.joblib
├── pipeline.joblib.sha256
├── metrics.json
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training the Model

Train the model with leakage-control mechanisms enabled:

```bash
python train_model.py \
    --real data/True.csv \
    --fake data/Fake.csv \
    --strip-source-artifacts \
    --eval-out-of-source
```

### Training Pipeline

The training process performs:

* Dataset loading
* Text preprocessing
* Deduplication
* Source leakage analysis
* Train/test splitting
* TF-IDF feature extraction
* Logistic Regression training
* Stratified K-Fold Cross-Validation
* Holdout test evaluation
* Out-of-source evaluation
* Diagnostic visualization
* Model serialization
* SHA-256 checksum generation

---

# 🔍 CLI Inference

Run inference on a headline or text snippet:

```bash
python detect_fake_news.py \
    --text "BREAKING: Scientists discover city on Mars" \
    --threshold 0.5 \
    --uncertainty-margin 0.10
```

The classifier produces a style-risk probability and uses the configured threshold to determine the prediction.

### Uncertainty Handling

Instead of blindly returning a binary prediction, the system can identify uncertain predictions.

For example, with:

```text
Threshold = 0.50
Uncertainty Margin = 0.10
```

the uncertainty region becomes approximately:

```text
0.45 ───────── 0.50 ───────── 0.55
       UNCERTAIN
```

This helps prevent overconfident predictions when the model is close to its decision boundary.

---

# 🌐 Streamlit Dashboard

Launch the interactive web application:

```bash
streamlit run app.py
```

The dashboard allows users to:

* Enter news headlines or text.
* Adjust the decision threshold.
* Configure the uncertainty margin.
* View the predicted style-risk probability.
* Interpret predictions interactively.

---

# 📈 Model Evaluation

The project generates multiple diagnostic artifacts to evaluate model performance.

### ROC Curve

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Precision-Recall Curve

Provides additional insight into classifier performance, particularly when class distributions are imbalanced.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Cross-Validation

Stratified K-Fold Cross-Validation is performed **only on the training data**, preventing information from the final test set from leaking into model development.

### Out-of-Source Evaluation

The project additionally performs group-based evaluation using `GroupShuffleSplit`.

Instead of randomly splitting individual articles, entire source/subject groups can be held out.

This provides a more realistic estimate of whether the model can generalize to **unseen sources**.

---

# 🔐 Model Integrity

Trained models are saved using Joblib together with a SHA-256 checksum:

```text
pipeline.joblib
pipeline.joblib.sha256
```

Before loading the model, the checksum can be verified to detect:

* Corrupted model files
* Unexpected modifications
* Incomplete transfers

The `model_compat.py` module also provides compatibility handling for differences between Scikit-Learn versions.

---

# 🧪 Reproducibility

The project is designed to make experiments reproducible by:

* Keeping preprocessing inside the Scikit-Learn pipeline.
* Separating training and evaluation data.
* Using deterministic random seeds where applicable.
* Saving trained model artifacts.
* Saving evaluation metrics in JSON format.
* Saving diagnostic plots.
* Recording the configuration used for experiments.

---

# 📦 Output Artifacts

After training, the project can generate:

```text
pipeline.joblib
pipeline.joblib.sha256
metrics.json
charts/
```

### `pipeline.joblib`

Serialized Scikit-Learn preprocessing and classification pipeline.

### `pipeline.joblib.sha256`

SHA-256 checksum used for artifact integrity verification.

### `metrics.json`

Stores model evaluation metrics and experiment results.

### `charts/`

Contains diagnostic plots such as:

* ROC Curve
* Precision-Recall Curve
* Confusion Matrix

---

# 💡 Why This Approach?

Traditional fake-news classifiers can produce impressive accuracy while learning **dataset-specific shortcuts** instead of genuine linguistic patterns.

This project therefore emphasizes:

```text
High Accuracy
      ↓
Is it actually generalizable?
      ↓
Check Dataset Leakage
      ↓
Check Source Confounding
      ↓
Use Proper Validation
      ↓
Evaluate on Unseen Sources
```

The goal is not simply to maximize accuracy, but to determine **how reliably the model generalizes beyond the patterns it encountered during training**.

---

# ⚠️ Important Limitation

This system should **not be treated as a factual truth detector**.

The model identifies **style-related risk patterns** learned from the training dataset. A prediction of `FAKE` does not prove that an article is factually false, and a prediction of `REAL` does not guarantee that the information is true.

The system should therefore be considered a **style-risk assessment tool**, not a replacement for professional fact-checking or source verification.

---

# 🔮 Future Improvements

Potential improvements include:

* Transformer-based NLP models such as BERT/RoBERTa.
* Explainable AI using SHAP or LIME.
* Real-world multi-source news datasets.
* Temporal validation using publication dates.
* Automated fact-checking through trusted sources.
* Model monitoring and drift detection.
* Docker-based deployment.
* REST API using FastAPI.
* Cloud deployment.
* Automated CI/CD model validation.

---

# 👨‍💻 Project Goal

The primary goal of this project is to demonstrate how a machine learning system can be designed with **data leakage awareness, robust validation, reproducibility, artifact integrity, and uncertainty-aware inference** rather than focusing solely on classification accuracy.

---

## 📜 License

Add your preferred license here, for example:

```text
MIT License
```

---

## ⭐ Acknowledgements

This project uses the ISOT Fake News Dataset for experimentation and machine learning development.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
