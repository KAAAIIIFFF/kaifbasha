import pandas as pd
from sklearn.datasets import fetch_openml

# Load Boston dataset from scikit-learn via OpenML
boston = fetch_openml(name="boston", version=1, as_frame=True)
df_sklearn = boston.frame

# Load Kaggle version
df_kaggle = pd.read_csv("BostonHousing.csv")

print("Scikit-learn dataset shape:")
print(df_sklearn.shape)

print("Kaggle dataset shape:")
print(df_kaggle.shape)

print("\nScikit-learn columns:")
print(df_sklearn.columns)

print("\nKaggle columns:")
print(df_kaggle.columns)

# Calculate mean for each column and compare
sklearn_means = df_sklearn.mean(numeric_only=True)
kaggle_means = df_kaggle.mean(numeric_only=True)

diff = pd.DataFrame({
    'sklearn_mean': sklearn_means,
    'kaggle_mean': kaggle_means
})
diff['difference'] = diff['sklearn_mean'] - diff['kaggle_mean']

# Save to CSV
diff.to_csv("boston_dataset_differences.csv")

print("\n✅ Comparison saved to 'boston_dataset_differences.csv'")


