import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("config/params.yaml"))["train"]

df = pd.read_csv("data/row.csv")
X = df.drop(["Fat(gm)", "calories(gm)"], axis=1)
y = df["calories(gm)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
