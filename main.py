# Preparing dataset

import pandas as pd

df = pd.read_csv("data/concrete.csv")
df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print("\nColumns: \n", df.columns)
print("\nFirst 5 rows: \n", df.head())
print("\nMissing values: \n", df.isnull().sum())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Progressive Model Implementation

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


models = {
    "1_Linear_Regression": LinearRegression(),

    "2_Polynomial_Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("linear", LinearRegression())
    ]),

    "3_Decision_Tree": DecisionTreeRegressor(random_state=42),

    "4_Random_Forest": RandomForestRegressor(random_state=42),

    "5_XGBoost": XGBRegressor(random_state=42, verbosity=0)
}


results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])


results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)

results_df = results_df.sort_values(by="R2 Score", ascending=False)

print("\nFinal Model Comparison:\n")
print(results_df)


results_df.to_csv("results.csv", index=False)
