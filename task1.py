import pandas as pd

file_path = 'iris.csv'
data = pd.read_csv(file_path)

print("Data loaded successfully.")
print(data.head())

print("\nCount of Null values in each column:")
print(data.isnull().sum())

rows_with_nulls = data[data.isnull().any(axis=1)]
print("\nRows with Null values:")
print(rows_with_nulls)
