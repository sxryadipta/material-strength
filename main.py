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
