import pandas as pd
import numpy as np

data = pd.read_csv("iris.csv")
data["species"] = data["species"].astype("category").cat.codes

X = data.drop("species", axis=1).values
y = data["species"].values

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

class NB:
  def fit(self, X, y):
        self.cls = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        for c in self.cls:
            xc = X[y == c]
            self.mean[c] = xc.mean(axis=0)
            self.var[c] = xc.var(axis=0)+1e-6
            self.prior[c] = xc.shape[0]/X.shape[0]

  def g(self, x, m, v):
     num = np.exp(-((x - m)**2)/(2*v))
     den = np.sqrt(2*np.pi*v)
     return num/den

  def predict(self, X):
       out = []
       for i in X:
        post = []
        for c in self.cls:
           p = np.log(self.prior[c])
           cond = np.sum(np.log(self.g(i, self.mean[c], self.var[c])))
           total = p + cond
           post.append(total)
        out.append(self.cls[np.argmax(post)])
       return np.array(out)

model = NB()
model.fit(Xtrain,Ytrain)
pred = model.predict(Xtest)

acc = np.mean(pred == Ytest)
print("acc is :", acc)
