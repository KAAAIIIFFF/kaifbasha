import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib

df = pd.read_csv("iris.csv")

le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

X = df[["petal_length", "petal_width"]]
y = df["species"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

xtrain , xtest , ytrain , ytest = train_test_split(X, y, test_size = 0.2 , random_state=42 , stratify=y)

param = {"n_neighbors": list(range(1, 11))}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param, cv=5)
grid.fit(xtrain, ytrain)

print("best k:", grid.best_params_)
print("best score:", grid.best_score_)

best_model = grid.best_estimator_
joblib.dump(best_model, "best_knn_model.joblib")
