#autolog when we using autolog so we dont need to use manualy log ex : artifacts, metric, params 

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub
import os

# 🔗 Dagshub remote MLflow setup
dagshub.init(repo_owner='mdakeel', repo_name='milk-calories-predictor-with-mlflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mdakeel/milk-calories-predictor-with-mlflow.mlflow")
mlflow.set_experiment("calories_prediction_exp_1")

# ✅ Enable autologging
mlflow.autolog()

# 📂 Load data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# 🚀 Train and evaluate in one run
with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    #  Log evaluation metrics manually (autolog doesn’t cover test metrics)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.set_tag("stage", "full_pipeline")


    print("Training + Evaluation complete")
    print("MSE:", mse, "R²:", r2)
    print(" Run URL:", f"https://dagshub.com/mdakeel/milk-calories-predictor-with-mlflow.mlflow/#/experiments/0/runs/{run.info.run_id}")
