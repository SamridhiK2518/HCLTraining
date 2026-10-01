import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
df = pd.read_csv("Housing Price Prediction/housing_anomalous_1000.csv")
print("Original dataset shape:");
df.shape;

df["Area_sqft"]=pd.to_numeric(df["Area_sqft"], errors="coerce");
df["Area_sqft"]=df["Area_sqft"].replace(0, np.nan);

df["Price_Lakhs"] = (
    df["Price_Lakhs"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)
df["Price_Lakhs"] = pd.to_numeric(df["Price_Lakhs"], errors="coerce");
df.loc[
    df["Price_Lakhs"] <= 0,
    "Price_Lakhs"
] = np.nan
df["Bedrooms"] = pd.to_numeric(df["Bedrooms"], errors="coerce");
df["Bedrooms"] = df["Bedrooms"].replace(0, np.nan);
df["Bathrooms"]=pd.to_numeric(df["Bathrooms"], errors="coerce");
df["Bathrooms"]=df["Bathrooms"].replace(0, np.nan);
df["Age_years"]=pd.to_numeric(df["Age_years"], errors="coerce");
df.loc[df["Age_years"] < 0, "Age_years"] = np.nan;
df["Parking_spaces"] = pd.to_numeric(
    df["Parking_spaces"], errors="coerce"
)

df.loc[
    df["Parking_spaces"] < 0,
    "Parking_spaces"
] = np.nan
df["Floor"] = pd.to_numeric(df["Floor"], errors="coerce")

df.loc[df["Floor"] < 0, "Floor"] = np.nan
df["Distance_to_Metro_km"] = pd.to_numeric(
    df["Distance_to_Metro_km"],
    errors="coerce"
)

# Location
df["Location"] = df["Location"].astype("object")
df["Location"] = df["Location"].apply(
    lambda x: str(x).strip() if pd.notna(x) else np.nan
)
df.loc[
    df["Location"].isin(["", "nan", "None"]),
    "Location"
] = np.nan


# Property Type
df["Property_Type"] = df["Property_Type"].astype("object")
df["Property_Type"] = df["Property_Type"].apply(
    lambda x: str(x).strip() if pd.notna(x) else np.nan
)
df.loc[
    df["Property_Type"].isin(["", "nan", "None"]),
    "Property_Type"
] = np.nan


# Furnishing
df["Furnishing"] = df["Furnishing"].astype("object")
df["Furnishing"] = df["Furnishing"].apply(
    lambda x: str(x).strip() if pd.notna(x) else np.nan
)
df.loc[
    df["Furnishing"].isin(["", "nan", "None"]),
    "Furnishing"
] = np.nan
Numeric_columns = [
    "Area_sqft",
    
    "Bedrooms", 
    "Bathrooms",
    "Age_years",
    "Parking_spaces",
    "Floor",
    "Distance_to_Metro_km"
]
Categorical_columns=[
    "Location",
    "Property_Type",
    "Furnishing"
]
# Remove rows where target price is missing
df = df.dropna(subset=["Price_Lakhs"])
y=df["Price_Lakhs"];
X=df.drop(columns=["Price_Lakhs"]);

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
imputer = IterativeImputer(
    max_iter=1000,
    random_state=42
)
X_train_numeric = imputer.fit_transform(
    X_train[Numeric_columns]
)
X_test_numeric = imputer.transform(
    X_test[Numeric_columns]
)
Categorical_columns = [
    "Location",
    "Property_Type",
    "Furnishing"
]
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat = cat_imputer.fit_transform(
    X_train[Categorical_columns]
)
X_test_cat = cat_imputer.transform(
    X_test[Categorical_columns]
)
X_train_numeric = pd.DataFrame(
    X_train_numeric,
    columns=Numeric_columns,
    index=X_train.index
)

X_test_numeric = pd.DataFrame(
    X_test_numeric,
    columns=Numeric_columns,
    index=X_test.index
)
outlier_columns = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_years",
    "Parking_spaces",
    "Floor",
    "Distance_to_Metro_km"
]

for col in outlier_columns:
    Q1 = X_train_numeric[col].quantile(0.25)
    Q3 = X_train_numeric[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(col, lower_bound, upper_bound)



    X_train_numeric[col] = X_train_numeric[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    X_test_numeric[col] = X_test_numeric[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )
Q1 = X_train_numeric[col].quantile(0.25)
Q3 = X_train_numeric[col].quantile(0.75)
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
X_train_cat_encoded = encoder.fit_transform(
    X_train[Categorical_columns]
)
X_test_cat_encoded = encoder.transform(
    X_test[Categorical_columns]
)
X_train_cat_encoded = pd.DataFrame(
    X_train_cat_encoded,
    columns=encoder.get_feature_names_out(Categorical_columns),
    index=X_train.index
)

X_test_cat_encoded = pd.DataFrame(
    X_test_cat_encoded,
    columns=encoder.get_feature_names_out(Categorical_columns),
    index=X_test.index
)
X_train_final = pd.concat(
    [X_train_numeric, X_train_cat_encoded],
    axis=1
)

X_test_final = pd.concat(
    [X_test_numeric, X_test_cat_encoded],
    axis=1
)
model = DecisionTreeRegressor(
    random_state=42
)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)