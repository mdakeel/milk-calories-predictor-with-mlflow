import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import os

#for dagshub remote mlflow server
import dagshub
dagshub.init(repo_owner='mdakeel', repo_name='milk-calories-predictor-with-mlflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/mdakeel/milk-calories-predictor-with-mlflow.mlflow")
mlflow.set_experiment("calories_prediction_exp_1")

# params = yaml.safe_load(open("config/params.yaml"))["train"]

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

os.makedirs("models", exist_ok=True)

with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    # mlflow.log_params(params)
    mlflow.set_tags({"Author": 'Aakil Tayyab', "Project" : 'Calories Preditor'})
    mlflow.sklearn.save_model(model, path="models/model")
    mlflow.log_artifacts("output/sklearn_model", artifact_path="model")


    print("Model saved with Run ID:", run.info.run_id)
