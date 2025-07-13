import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

print("Naive Bayes Accuracy:", nb_acc)
print("Logistic Regression Accuracy:", log_acc)

df_subset = df[["sepal_length", "sepal_width", "species"]]
X2 = df_subset.drop("species", axis=1)
y2 = df_subset["species"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)

model2 = LogisticRegression()
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)

print("Logistic Regression Accuracy (2 features):", accuracy_score(y2_test, pred2))

df_test = pd.DataFrame(X_test, columns=df.columns[:-1])
df_test["Predicted"] = log_pred

sns.pairplot(df_test.assign(Species=log_pred), hue="Species", diag_kind="hist")
plt.suptitle("Pairplot Colored by Logistic Regression Predictions", y=1.02)
plt.show()

