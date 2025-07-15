import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("iris.csv")

le = LabelEncoder()
data["species"] = le.fit_transform(data["species"])

X = data.drop("species", axis=1)
y = data["species"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

Xtrain , Xtest , Ytrain , Ytest = train_test_split(X , y , test_size = 0.2 , random_state=42 , stratify=y)

print("Xtrain shape :", Xtrain.shape)
print("Xtest shape :", Xtest.shape)
