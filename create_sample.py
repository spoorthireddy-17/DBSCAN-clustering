import pandas as pd

df = pd.read_csv("train.csv")
df_sample = df.sample(n=50000, random_state=42)
df_sample.to_csv("train_sample.csv", index=False)

print("Sample file created successfully!")
