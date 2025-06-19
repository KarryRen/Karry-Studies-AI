# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 16:53
# @Author  : Karry Ren

""" The ML network. """

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


def random_search(seed: int) -> RandomizedSearchCV:
    """ Define the random search model.

    :param seed: Random seed.
    :return: A set RandomizedSearchCV.

    """

    # ---- Param dist ---- #
    param_dist = {
        "n_estimators": [200, 250, 300],
        "learning_rate": [0.05, 0.08, 0.1],
        "max_depth": [3, 4],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "gamma": [0, 0.05, 0.1],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0.1, 0.5, 1]
    }

    # ---- XGB model ---- #
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        early_stopping_rounds=10
    )

    # ---- Random search ---- #
    rs = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=1,
        scoring="neg_root_mean_squared_error",
        cv=2,
        verbose=2,
        n_jobs=10,
        random_state=seed
    )
    return rs
