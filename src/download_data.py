import pandas as pd

url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"

df = pd.read_csv(url)

df.to_csv("data/raw/heart.csv", index=False)

print("Dataset downloaded successfully!")