import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("Customer-Churn.xls")

# customerID is a unique identifier with no predictive value — drop it
df = df.drop(columns="customerID")

#  Convert it to a proper numeric column BEFORE the label-encoding

df["TotalCharges"] = df["TotalCharges"].replace(" ", "0").astype(float)

# Encode all remaining categorical columns
object_columns = df.select_dtypes(include="object").columns.tolist()
object_columns = ["SeniorCitizen"] + object_columns

# Dictionary to save the fitted encoders — needed later for predict_churn()
encoders = {}
for column in object_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

with open("encoders.joblib", "wb") as f:
    joblib.dump(encoders, f)

x = df.drop(columns="Churn")
y = df["Churn"]
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Applying SMOTE to balance the training set
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
print(y_train_smote.value_counts())

# Compare a few candidate models via cross-validation
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
}
cv_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name} with default parameters")
    scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} cross validation accuracy: {np.mean(scores):.2f}")
    print("-" * 50)

# Final model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train_smote, y_train_smote)

# Evaluate on the held-out test set
y_test_pred = rfc.predict(x_test)
print(f"Accuracy Score: {accuracy_score(y_test_pred, y_test)}")
print(f"Confusion matrix: {confusion_matrix(y_test_pred, y_test)}")
print(f"Classification Report: {classification_report(y_test_pred, y_test)}")

# Save the model + the exact feature order it expects
model_data = {"model": rfc, "feature_names": x.columns.to_list()}
with open("customer_churn_model.joblib", "wb") as f:
    joblib.dump(model_data, f)

with open("customer_churn_model.joblib", "rb") as f:
    model_data = joblib.load(f)

loaded_model = model_data["model"]
feature_names = model_data["feature_names"]


def predict_churn(customer_data):
    input_df = pd.DataFrame([customer_data])

    if "TotalCharges" in input_df.columns:
        input_df["TotalCharges"] = input_df["TotalCharges"].replace({" ": 0.0}).astype(float)

    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    input_df = input_df[feature_names]
    prediction = loaded_model.predict(input_df)[0]

    if prediction == 1:
        print("The customer is likely to CHURN!")
    else:
        print("The customer is likely to STAY.")


new_customer = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7, "TotalCharges": "151.65"
}

predict_churn(new_customer)