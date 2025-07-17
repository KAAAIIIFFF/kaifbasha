import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("iris.csv")

le = LabelEncoder()
data["species"] = le.fit_transform(data["species"])

X = data.drop("species", axis=1)
y = data["species"]

sc = MinMaxScaler()
X = sc.fit_transform(X)

Xtrain , Xtest , Ytrain , Ytest = train_test_split(X , y , test_size = 0.2 , random_state=42 , stratify=y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(Xtrain , Ytrain)
pred = model.predict(Xtest)

acc = accuracy_score(Ytest , pred)
print("accuracy is :", acc)

cm = confusion_matrix(Ytest, pred)
print("confusion matrix:\n", cm)

sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

k_range = range(1,11)
accs = []

for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(Xtrain, Ytrain)
  p = knn.predict(Xtest)
  a = accuracy_score(Ytest, p)
  accs.append(a)

plt.plot(k_range, accs, marker='o')
plt.title("Accuracy vs K")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
