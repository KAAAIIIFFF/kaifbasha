import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("BostonHousingData.csv")


corr = df.corr()


print("Correlation Matrix:")
print(corr)


plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


target_corr = corr["medv"].sort_values(ascending=False)

print("\nCorrelation of each feature with 'medv':")
print(target_corr)
