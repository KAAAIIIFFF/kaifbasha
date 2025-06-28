import pandas as pd

file_path = 'iris.csv'
data = pd.read_csv(file_path)

print("Summary statistics:")
print(data.describe())

print("\nSamples per species:")
print(data['species'].value_counts())

filtered_data = data[data['petal_length'] > 1.5]
print("\nRows where petal length > 1.5:")
print(filtered_data)

data['species_encoded'] = data['species'].astype('category').cat.codes
print("\nSpecies encoded:")
print(data[['species', 'species_encoded']])

data['petal_ratio'] = data['petal_length'] / data['petal_width']
print("\nData with petal_ratio column:")
print(data[['petal_length', 'petal_width', 'petal_ratio']])
