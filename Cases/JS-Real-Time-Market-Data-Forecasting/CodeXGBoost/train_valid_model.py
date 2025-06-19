# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 16:51
# @Author  : Karry Ren

""" Train and validate models. """

import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from joblib import parallel_backend

from models.net import random_search
import config as config
from utils import fix_random_seed


def train_valid_model():
    """ Train & Valid Model. """

    logging.info(f"***************** RUN TRAIN&VALID MODEL *****************")

    # ---- Load the data ---- #
    logging.info(f"**** TRAINING & VALID DATA ****")
    data_train = np.load("../Data/dataset_xgboost/train.npz")
    X_train, y_train, w_train = data_train["x"].astype(np.float16), data_train["y"].astype(np.float16), data_train["w"].astype(np.float16)
    X_train = X_train.reshape(X_train.shape[0], -1)
    logging.info(f"train dataset: x_train.shape={X_train.shape}, y_train.shape={y_train.shape}, w_train.shape={w_train.shape}")
    data_valid = np.load("../Data/dataset_xgboost/valid.npz")
    X_valid, y_valid, w_valid = data_valid["x"].astype(np.float16), data_valid["y"].astype(np.float16), data_valid["w"].astype(np.float16)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    logging.info(f"valid dataset: x_valid.shape={X_valid.shape}, y_valid.shape={y_valid.shape}, w_valid.shape={w_valid.shape}")

    # ---- Construct the model and transfer device ---- #
    model = random_search(config.RANDOM_SEED)
    with parallel_backend("threading", n_jobs=4):
        model.fit(
            X_train, y_train, sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
    best_model = model.best_estimator_
    logging.info(f"Model training is over.")

    # ---- Metrics eval ---- #
    y_pred = best_model.predict(X_valid)
    y_valid, w_valid = y_valid[:, 0], w_valid[:, 0]
    standard_rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))
    weighted_rmse = np.sqrt(np.mean(w_valid * (y_valid - y_pred) ** 2))
    upward_error = np.mean(np.abs(y_pred[y_valid >= 0] - y_valid[y_valid >= 0]))
    downward_error = np.mean(np.abs(y_pred[y_valid < 0] - y_valid[y_valid < 0]))
    logging.info("\n=== Final Results ===")
    logging.info(f"***** Best Parameters: {model.best_params_}")
    logging.info(f"***** Standard RMSE: {standard_rmse:.4f}")
    logging.info(f"***** Weighted RMSE: {weighted_rmse:.4f}")
    logging.info(f"***** Upside Error: {upward_error:.4f}")
    logging.info(f"***** Downside Error: {downward_error:.4f}")
    logging.info(f"***** Downside Reduction: {(1 - downward_error / upward_error):.2%}")

    # ---- Visual ---- #
    plt.figure(figsize=(15, 5))
    # error distribution
    plt.subplot(1, 3, 1)
    plt.hist(y_pred[y_valid < 0] - y_valid[y_valid < 0], bins=50, alpha=0.7, label="Downside")
    plt.hist(y_pred[y_valid >= 0] - y_valid[y_valid >= 0], bins=50, alpha=0.7, label="Upside")
    plt.legend()
    plt.title("Error Distribution")
    # pred vs. actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_valid, y_pred, alpha=0.1)
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], "r--")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Prediction vs Actual")
    # gap error
    plt.subplot(1, 3, 3)
    quantiles = np.linspace(0, 1, 11)
    q_errors = [np.quantile(np.abs(y_valid - y_pred), q) for q in quantiles]
    plt.plot(quantiles, q_errors, "bo-")
    plt.xlabel("Quantile")
    plt.ylabel("Absolute Error")
    plt.title("Quantile Error Analysis")
    plt.tight_layout()
    # save
    plt.savefig(f"{config.SAVE_PATH}/complex1.png", dpi=300)

    # ---- Feature Importance ---- #
    plt.figure(figsize=(10, 6))
    sorted_idx = best_model.feature_importances_.argsort()
    plt.barh(np.array(range(64 * 3))[sorted_idx], best_model.feature_importances_[sorted_idx])
    plt.title("Feature Importance")
    plt.savefig(f"{config.SAVE_PATH}/feature_importance.png", dpi=300)


if __name__ == "__main__":
    # ---- Prepare some environments for training and prediction ---- #
    # fix the random seed
    fix_random_seed(seed=config.RANDOM_SEED)
    # build up the save directory
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    # construct the train&valid log file
    logging.basicConfig(filename=config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # ---- Train & Valid model ---- #
    train_valid_model()
