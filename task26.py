import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("iris.csv")

X = df[["petal_length", "petal_width"]]

scaler = joblib.load("knn_scaler.joblib")
X_scaled = scaler.transform(X)

model = joblib.load("best_knn_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

preds = model.predict(X_scaled)

species_names = label_encoder.inverse_transform(preds)

df["Predicted_Species"] = species_names

df.to_csv("knn_predictions.csv", index=False)

print("Predictions saved to knn_predictions.csv")

