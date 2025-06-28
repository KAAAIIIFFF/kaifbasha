import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('iris.csv')

plt.hist(data['sepal_length'])
plt.xlabel('Sepal Length')
plt.ylabel('Count')
plt.title('Sepal Length Histogram')
plt.show()

plt.scatter(data['sepal_length'], data['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')
plt.show()

sns.pairplot(data, hue='species')
plt.show()

sns.boxplot(x='species', y='petal_width', data=data)
plt.title('Petal Width by Species')
plt.show()
