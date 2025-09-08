import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import yaml

params = yaml.safe_load(open("config/params.yaml"))["train"]

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

mlflow.set_experiment("calories_prediction_exp_1")

with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    mlflow.log_params(params)
    mlflow.log_artifact(__file__)
    mlflow.log_artifacts("output/")
    mlflow.set_tags({"Author": 'Aakil Tayyab', "Project" : 'Calories Preditor'})
    mlflow.sklearn.log_model(model, "model")

    print("Model saved with Run ID:", run.info.run_id)
