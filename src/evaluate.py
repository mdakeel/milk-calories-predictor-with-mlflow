from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import pandas as pd
import yaml

import dagshub
dagshub.init(repo_owner='mdakeel', repo_name='milk-calories-predictor-with-mlflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/mdakeel/milk-calories-predictor-with-mlflow.mlflow")
# mlflow.set_experiment("calories_prediction_exp_1")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

model = mlflow.sklearn.load_model("mlflow-artifacts:/c8276d2f1cab432ab01384b2895441fa/af9e5daf0f0f4d009e584187d5fe9900/artifacts/model")
preds = model.predict(X_test)

params = yaml.safe_load(open("config/params.yaml"))["train"]

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

with mlflow.start_run() as run:
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_params(params)

print("✅ Evaluation complete. MSE:", mse, "R²:", r2)
print("🔗 Evaluation run:", run.info.run_id)
