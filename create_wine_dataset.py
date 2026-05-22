import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)

df = wine.frame.copy()

df = df.drop(columns=["target"])

df.insert(0, "Wine_ID", [f"Wine_{i+1:03d}" for i in range(len(df))])

df.to_csv("wine_dataset_for_nmf_webapp.csv", index=False)

print("Created: wine_dataset_for_nmf_webapp.csv")
print(df.head())