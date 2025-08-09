import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/california_housing_processed.csv")


def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["MedHouseVal"]).copy()
    y = df["MedHouseVal"].copy()
    return X, y

def eval_and_log(y_true, y_pred, prefix=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    mlflow.log_metric(f"{prefix}mse", mse)
    mlflow.log_metric(f"{prefix}rmse", rmse)
    mlflow.log_metric(f"{prefix}r2", r2)
    return dict(mse=mse, rmse=rmse, r2=r2)


def train_and_log(model, model_name):
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_name", model_name)
        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", getattr(model, "max_depth"))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} → RMSE: {rmse:.4f}, R2: {r2:.4f}")

def main(args):
    data_path = Path(args.data)
    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )

    mlflow.set_experiment(args.experiment_name)

    # 1) Linear Regression
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        mlflow.log_param("model", "LinearRegression")
        eval_and_log(y_test, preds)
        mlflow.sklearn.log_model(lr, "model")

    # 2) Decision Tree
    with mlflow.start_run(run_name="DecisionTree"):
        max_depth = args.dt_max_depth
        dt = DecisionTreeRegressor(max_depth=max_depth, random_state=args.random_seed)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_test)
        mlflow.log_param("model", "DecisionTree")
        mlflow.log_param("dt_max_depth", max_depth)
        eval_and_log(y_test, preds)
        mlflow.sklearn.log_model(dt, "model")

    print("Training complete. Check MLflow UI for runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/california_housing_processed.csv")
    parser.add_argument("--experiment-name", type=str, default="california-housing")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dt-max-depth", type=int, default=8)
    args = parser.parse_args()
    main(args)