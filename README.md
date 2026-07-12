# Customer Churn Prediction

A machine learning model that predicts whether a telecom customer is likely to churn (cancel their subscription), based on their account details, services used, and billing information.

## Overview

Customer churn is one of the most business-critical prediction problems in subscription-based industries — identifying at-risk customers early lets a company intervene before losing them. This project trains a classifier on real telecom customer data (demographics, subscribed services, contract type, and billing) to predict churn risk, and includes a ready-to-use function for scoring a single new customer.

## How It Works

1. **Data** — Used a real telecom customer dataset (7,043 customers, 21 original columns) covering demographics, subscribed services (phone, internet, streaming, etc.), contract type, and billing details, with `Churn` as the target.
2. **Cleaning** — Dropped `customerID` (a unique identifier with no predictive value), and converted `TotalCharges` from text to a proper numeric column, handling 11 blank entries by treating them as 0.
3. **Encoding** — Label-encoded all categorical columns (gender, contract type, internet service, etc.) and saved the fitted encoders so new customer data can be transformed consistently at prediction time.
4. **Handling Class Imbalance** — Applied **SMOTE** to the training set, since churned customers are a minority class in the raw data.
5. **Model Comparison** — Cross-validated Decision Tree, Random Forest, and XGBoost classifiers:

   | Model | Cross-Validated Accuracy |
   |---|---|
   | Decision Tree | 79% |
   | **Random Forest** | **84%** (best) |
   | XGBoost | 83% |

6. **Final Evaluation** — The selected Random Forest model reached **77.9% accuracy** on the held-out test set, with stronger performance on predicting "stay" (85% precision/recall) than "churn" (58% precision/recall) — a realistic outcome given churn is the harder, less common class to predict correctly.
7. **Prediction Function** — Included a `predict_churn()` function that takes a single customer's raw details (as a plain dictionary) and returns a clear churn/stay prediction, handling all the same encoding and cleaning steps used during training.

## Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Modeling:** Scikit-learn (Random Forest, Decision Tree), XGBoost
- **Class Balancing:** imbalanced-learn (SMOTE)
- **Visualization (EDA):** Matplotlib, Seaborn

## Project Structure

```
customer-churn-prediction/
├── main.py                        # Data cleaning, encoding, model training & evaluation
├── customer_churn_model.joblib    # Trained Random Forest model + feature order
├── encoders.joblib                # Saved label encoders for categorical features
├── Customer-Churn.xls             # Raw dataset (7,043 customers)
└── requirements.txt               # Python dependencies
```

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/AyaanHussain1/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run training and see a sample prediction
python main.py
```

To predict a new customer, call `predict_churn()` with a dictionary of their details (see the `new_customer` example at the bottom of `main.py`).

## Results

| Metric | Score |
|---|---|
| Cross-validated accuracy (Random Forest) | 84% |
| Test accuracy | 77.9% |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Stay (0) | 0.85 | 0.85 | 0.85 |
| Churn (1) | 0.58 | 0.58 | 0.58 |

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

This project is built for educational and portfolio purposes, using a publicly available telecom customer churn dataset.
