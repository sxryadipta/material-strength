import pandas as pd

df = pd.read_csv("data/concrete.csv")
df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print("\nColumns: \n", df.columns)
print("\nFirst 5 rows: \n", df.head())
print("\nMissing values: \n", df.isnull().sum())