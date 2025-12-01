# Telco Customer Churn Prediction

Customer Churn Prediction using the Telco dataset. This project builds an end-to-end ML pipeline with data cleaning, EDA, feature engineering, and model training using Logistic Regression. Includes handling categorical data, class imbalance, and model evaluation (AUC, F1).

## Project Structure

```
telco-churn-ml/
├── data/
│   ├── raw/               # Original raw datasets
│   ├── processed/         # Cleaned/processed datasets
│   └── external/          # Additional datasets or lookups
├── notebooks/
│   └── 01_churn_model.ipynb   # EDA + model prototyping
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and prediction
│   └── evaluation/        # Metrics and evaluation
├── tests/                 # Unit tests
├── models/                # Saved model artifacts
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training a model

```python
from src.data import load_raw_data, clean_data, split_data
from src.models import train_pipeline, save_model

# Load and prepare data
df = load_raw_data()
df = clean_data(df)
X_train, X_test, y_train, y_test = split_data(df)

# Train and save model
pipeline = train_pipeline(X_train, y_train)
save_model(pipeline)
```

### Making predictions

```python
from src.models import load_model, predict_proba

model = load_model()
probabilities = predict_proba(model, X_test)
```

### Evaluation

```python
from src.evaluation import evaluate_model, get_classification_report

y_pred = model.predict(X_test)
y_proba = predict_proba(model, X_test)

metrics = evaluate_model(y_test, y_pred, y_proba)
print(get_classification_report(y_test, y_pred))
```

## Running Tests

```bash
pytest tests/
```

## Model Performance

- ROC AUC Score: ~0.84
