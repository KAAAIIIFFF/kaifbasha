import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib

df = pd.read_csv("iris.csv")

le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)

joblib.dump(model, "naive_bayes_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")

def predict_species(features):
    model = joblib.load("naive_bayes_model.joblib")
    scaler = joblib.load("scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df_input = pd.DataFrame([features], columns=cols)
    features_scaled = scaler.transform(df_input)
    prediction = model.predict(features_scaled)
    return le.inverse_transform(prediction)[0]


results = []
for features in X_test:
    species = predict_species(features)
    results.append(species)

df_results = pd.DataFrame(X_test, columns=X.columns)
df_results["Predicted Species"] = results
df_results.to_csv("naive_bayes_predictions.csv", index=False)

