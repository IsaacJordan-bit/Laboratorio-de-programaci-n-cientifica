"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""
import logging
from typing import Dict
from lightgbm import LGBMRegressor

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
import numpy as np


def split_data(data: pd.DataFrame, params: Dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_mae")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def train_model(X_train, y_train,  X_valid, y_valid):
    

    id = mlflow.create_experiment('Experiment_1')
    mlflow.autolog() # registrar automáticamente información del entrenamiento   

    rf = RandomForestRegressor()
    with mlflow.start_run(run_name = 'rf'): # delimita inicio y fin del run
        rf.fit(X_train, y_train) # train the model
        predictions = np.array(rf.predict(X_valid)) # Use the model to make predictions on the test dataset.
        true_labels = np.array(y_valid)
        valid_mae = np.mean(np.abs(predictions-true_labels))
        mlflow.log_metric('valid_mae', valid_mae)

    lr = LinearRegression()
    with mlflow.start_run(run_name = 'lr'): # delimita inicio y fin del run
        lr.fit(X_train, y_train) # train the model
        predictions = np.array(lr.predict(X_valid)) # Use the model to make predictions on the test dataset.
        true_labels = np.array(y_valid)
        valid_mae = np.mean(np.abs(predictions-true_labels))
        mlflow.log_metric('valid_mae', valid_mae)

    svr = SVR()
    with mlflow.start_run(run_name = 'svr'): # delimita inicio y fin del run
        svr.fit(X_train, y_train) # train the model
        predictions = np.array(svr.predict(X_valid)) # Use the model to make predictions on the test dataset.
        true_labels = np.array(y_valid)
        valid_mae = np.mean(np.abs(predictions-true_labels))
        mlflow.log_metric('valid_mae', valid_mae)

    xgb = XGBRegressor()
    with mlflow.start_run(run_name = 'xgb'): # delimita inicio y fin del run
        xgb.fit(X_train, y_train) # train the model
        predictions = np.array(xgb.predict(X_valid)) # Use the model to make predictions on the test dataset.
        true_labels = np.array(y_valid)
        valid_mae = np.mean(np.abs(predictions-true_labels))
        mlflow.log_metric('valid_mae', valid_mae)

    lgbmr = LGBMRegressor()
    with mlflow.start_run(run_name = 'lgbmr'): # delimita inicio y fin del run
        lgbmr.fit(X_train, y_train) # train the model
        predictions = np.array(lgbmr.predict(X_valid)) # Use the model to make predictions on the test dataset.
        true_labels = np.array(y_valid)
        valid_mae = np.mean(np.abs(predictions-true_labels))
        mlflow.log_metric('valid_mae', valid_mae)

    return get_best_model(id)








def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Mean Absolute Error of {mae} on test data.")
