import pandas as pd

df = pd.read_csv("iris.csv")

print("Shape of dataset:", df.shape)
print("Unique species:", df["species"].unique())
