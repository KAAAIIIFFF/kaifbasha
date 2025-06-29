import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib

iris = load_iris(as_frame=True)
df = pd.concat([iris.data, iris.target], axis=1)
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']

scaler = MinMaxScaler()
df[['sepal_length','sepal_width','petal_length','petal_width']] = scaler.fit_transform(df[['sepal_length','sepal_width','petal_length','petal_width']])

df.to_csv('iris_cleaned.csv', index=False)

X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['species']

model = LogisticRegression(max_iter=200)
model.fit(X, y)

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    vals = [[sepal_length, sepal_width, petal_length, petal_width]]
    vals_scaled = scaler.transform(vals)
    pred = model.predict(vals_scaled)
    return pred[0]

counts = df['species'].value_counts()
plt.bar(counts.index, counts.values)
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Species Count')
plt.show()

joblib.dump(model, 'logistic_model.joblib')
