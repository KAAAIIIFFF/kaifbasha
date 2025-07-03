import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_csv("BostonHousingData.csv")


print("Missing values before filling:")
print(df.isnull().sum())


df = df.fillna(df.mean())

print("Missing values after filling:")
print(df.isnull().sum())


df.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10), sharex=False, sharey=False)
plt.tight_layout()
plt.show()


scaler_standard = StandardScaler()
scaled_standard = scaler_standard.fit_transform(df)

df_standard = pd.DataFrame(scaled_standard, columns=df.columns)


scaler_minmax = MinMaxScaler()
scaled_minmax = scaler_minmax.fit_transform(df)

df_minmax = pd.DataFrame(scaled_minmax, columns=df.columns)


print("\nStandard Scaler Means (should be ~0):")
print(df_standard.mean())

print("\nStandard Scaler Std Devs (should be ~1):")
print(df_standard.std())

print("\nMinMax Scaler Min values (should be 0):")
print(df_minmax.min())

print("\nMinMax Scaler Max values (should be 1):")
print(df_minmax.max())
