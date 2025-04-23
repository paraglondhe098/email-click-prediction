# 📧 Email Click Prediction

A machine learning system that predicts user engagement with marketing emails to optimize campaign targeting and increase click-through rates.

## 🎯 Objective

Help the marketing team send emails strategically by predicting the likelihood of users clicking on email links. This data-driven approach replaces random email distribution to maximize engagement and conversion rates.


## 📊 Data Overview

The model uses historical email interaction data including:
- User demographics and behavior
- Email content and design features
- Temporal factors (timing of emails)
- Previous engagement metrics

## 🧾 Features

Key predictive features in the final model:

```
['hour_sin', 'hour_cos', 'weekday', 'user_past_purchases',
 'email_text_short_email', 'email_version_personalized',
 'user_country_FR', 'user_country_UK', 'user_country_US']
```

Features were selected through importance analysis and include:
- Cyclical time encoding (sin/cos transformations)
- One-hot encoded categorical variables
- Historical user behavior metrics

## 🧪 Model Architecture

We implemented a **two-stage cascading model** to address the sequential nature of email engagement:

1. **Stage 1: Open Prediction**
   - **Algorithm**: CatBoost classifier
   - **Purpose**: Predicts probability of email being opened (`opened_proba`)
   - **Key feature**: Handles categorical features efficiently

2. **Stage 2: Click Prediction**
   - **Algorithm**: Logistic Regression
   - **Input**: Original features + `opened_proba` from Stage 1
   - **Purpose**: Predicts final probability of link click

This approach models the natural funnel (receive → open → click) and addresses the class imbalance by focusing on high-recall predictions.

## 📈 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Recall | 0.768 | Captures 76.8% of actual clickers |
| Precision | 0.044 | 4.4% of predicted clickers actually click |
| Accuracy | 0.616 | Overall prediction accuracy |
| F1 Score | 0.084 | Harmonic mean of precision and recall |
| AUC Score | 0.759 | Model's ability to distinguish between classes |

The model prioritizes recall over precision to ensure we identify as many potential clickers as possible, which is appropriate for email marketing where the cost of sending an additional email is low.

## 🧱 Project Structure

```
📁 artifacts/
    └── best_final.pkl           # Serialized final model

📁 src/
    ├── email_click_predictor.py # Custom model class implementation
    └── preprocessing.py         # Data processing pipeline

📁 notebooks/
    ├── exploration.ipynb        # EDA and feature analysis
    └── modeling.ipynb           # Model development and evaluation

📄 README.md                     # Project documentation
📄 requirements.txt              # Dependencies
```

## 💾 Usage Guide

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/email-click-prediction.git
cd email-click-prediction

# Set up environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### Making Predictions

```python
import pickle
import pandas as pd
from src.preprocessing import preprocess_data

# Load model
with open("artifacts/best_final.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare your data
new_data = pd.read_csv("path/to/your/data.csv")
X_processed = preprocess_data(new_data)

# Get click probabilities
click_probabilities = model.predict_proba(X_processed)[:, 1]

# Get binary predictions (using default threshold of 0.5)
predictions = model.predict(X_processed)
```

## ✅ Future Improvements

1. **Advanced Modeling**
   - Implement sequence models (LSTM/GRU) to capture temporal patterns in user engagement
   - Explore ensemble methods to improve predictive performance

2. **Feature Engineering**
   - Extract semantic features from email content using NLP
   - Incorporate user segmentation based on behavioral clusters

3. **Operational Enhancements**
   - Develop an A/B testing framework to validate model recommendations
   - Implement custom thresholding based on business objectives
   - Create a real-time prediction API for integration with email marketing platforms

4. **Uplift Modeling**
   - Shift from click prediction to incremental response modeling
   - Estimate causal impact of sending vs. not sending emails
