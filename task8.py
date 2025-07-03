import pandas as pd

df = pd.read_csv("BostonHousingData.csv")

print("Describe:")
print(df.describe())

print("\nInfo:")
print(df.info())

print("\nNull values in each column:")
print(df.isnull().sum())
