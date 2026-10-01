import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

df = pd.read_csv(
    "loan_defaulter_1000_rows_with_name_anomalies.csv"
)

print("Original dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())




print("\nMissing values:")
print(df.isnull().sum())


print("\nDefault distribution:")
print(df["default"].value_counts())


df = df.drop(
    columns=[
        "customer_id",
        "name",
        "anomaly_type",
        "anomaly_marker"
    ],
    errors="ignore"
)



df["default"] = df["default"].map({
    "Yes": 1,
    "No": 0
})


##Cleaning and preprocessing the data

df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)


df.loc[
    (df["age"] < 18) | (df["age"] > 100),
    "age"
] = np.nan




df["monthly_income"] = (
    df["monthly_income"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["monthly_income"] = pd.to_numeric(
    df["monthly_income"],
    errors="coerce"
)


df.loc[
    df["monthly_income"] <= 0,
    "monthly_income"
] = np.nan




df["loan_amount"] = (
    df["loan_amount"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["loan_amount"] = pd.to_numeric(
    df["loan_amount"],
    errors="coerce"
)

df.loc[
    df["loan_amount"] <= 0,
    "loan_amount"
] = np.nan


df["credit_score"] = pd.to_numeric(
    df["credit_score"],
    errors="coerce"
)

df.loc[
    (df["credit_score"] < 300) |
    (df["credit_score"] > 900),
    "credit_score"
] = np.nan




df["employment_type"] = (
    df["employment_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["employment_type"] = df["employment_type"].replace({
    "self employed": "self-employed",
    "self-employed": "self-employed",
    "salaried": "salaried",
    "business": "business"
})




df["city"] = (
    df["city"]
    .astype(str)
    .str.strip()
    .str.lower()
)


df["previous_default"] = (
    df["previous_default"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["previous_default"] = df["previous_default"].replace({
    "y": "yes",
    "yes": "yes",
    "1": "yes",
    "n": "no",
    "no": "no",
    "0": "no"
})


df["loan_date"] = pd.to_datetime(
    df["loan_date"],
    errors="coerce",
    format="mixed"
)

df["loan_year"] = df["loan_date"].dt.year
df["loan_month"] = df["loan_date"].dt.month


df = df.drop(
    columns=["loan_date"]
)



before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicate rows removed:", before - after)
print("Rows after duplicate removal:", after)


X = df.drop(
    columns=["default"]
)

y = df["default"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))



print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


numerical_features = [
    "age",
    "monthly_income",
    "loan_amount",
    "credit_score",
    "loan_year",
    "loan_month"
]


categorical_features = [
    "employment_type",
    "city",
    "previous_default"
]



numeric_pipeline = Pipeline([
    
    # Missing numerical values
    # are replaced by training median
    ("imputer", SimpleImputer(
        strategy="median"
    )),

    # Standardization
    ("scaler", StandardScaler())
])



categorical_pipeline = Pipeline([

    # Missing categories
    # are replaced by most frequent category
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),

    # Convert categories to numbers
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])




preprocessor = ColumnTransformer([

    (
        "numerical",
        numeric_pipeline,
        numerical_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])



model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


pipeline = Pipeline([

    (
        "preprocessing",
        preprocessor
    ),

    (
        "model",
        model
    )
])


print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


y_pred = pipeline.predict(
    X_test
)



print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)


sensitivity = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

print("\nSensitivity (Recall) for Defaulters:")
print(sensitivity)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Defaulter",
            "Defaulter"
        ]
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("\nAUC:")
print(auc_score)
plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {auc_score:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve - Loan Defaulter Prediction")
plt.legend()
plt.grid()

plt.show()



print("\nConfusion Matrix Explanation:")

print("TN =", cm[0][0],
      "→ Correctly predicted Non-Defaulters")

print("FP =", cm[0][1],
      "→ Non-Defaulters incorrectly predicted as Defaulters")

print("FN =", cm[1][0],
      "→ Defaulters incorrectly predicted as Non-Defaulters")

print("TP =", cm[1][1],
      "→ Correctly predicted Defaulters")