from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df = pd.read_csv("titanic.csv")

# Drop rows with missing Age
df = df.dropna(subset=["Age"])

# Encode 'Sex'
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Features and label
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Optional: test input
while True:
    try:
        pclass = int(input("Enter class (1–3): "))
        sex = input("Enter sex (male/female): ")
        age = float(input("Enter age: "))
        fare = float(input("Enter fare: "))
        sex_encoded = 1 if sex.lower() == "male" else 0
        sample = scaler.transform([[pclass, sex_encoded, age, fare]])
        pred = model.predict(sample)[0]
        print("Prediction:", "Survived" if pred == 1 else "Did NOT survive")
    except:
        print("Exiting or input error.")
        break


