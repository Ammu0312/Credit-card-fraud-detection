import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib

# Load dataset
df = pd.read_csv("creditcard.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale ONLY Time & Amount (important)
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
#pickle.dump(model, open("model.pkl", "wb"))
#pickle.dump(scaler, open("scaler.pkl", "wb"))
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅, Model trained and saved!")