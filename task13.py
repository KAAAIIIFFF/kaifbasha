import pandas as pd
import numpy as np

df = pd.read_csv("BostonHousingData.csv")
df = df.fillna(df.mean())

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X = X.values
y = y.values

X = np.c_[np.ones(X.shape[0]), X]

beta = np.linalg.inv(X.T @ X) @ X.T @ y

y_pred = X @ beta

mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
ss_tot = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("Coefficients:", beta)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)
