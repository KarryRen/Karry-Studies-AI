# -*- coding: utf-8 -*-
# @Time    : 2025/6/4 10:16
# @Author  : Karry Ren

""" Use model to do the prediction. """

import logging
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from utils import load_best_model
from models.metrics import r2_score, accuracy_score, f1_score
from datasets.jsmp_dataset import JSMPDataset
import config as config


def pred_model() -> None:
    """ Pred Model. """

    logging.info(f"***************** RUN PRED MODEL *****************")

    # ---- Get the pred device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** IN DEVICE `{device}` *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"**** TEST DATALOADER ****")
    test_dataset = JSMPDataset(root_path=config.JSMP_DATASET_PATH, data_type="valid")
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the test dataloader
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Test dataset: length = {len(test_dataset)}")

    # ---- Load model and test ---- #
    model, model_path = load_best_model(config.MODEL_SAVE_PATH)
    logging.info(f"***************** LOAD Best Model {model_path} *****************")

    # ---- Test Model ---- #
    logging.info(f"***************** BEGIN TEST THE MODEL ! *****************")
    test_dataset_len = len(test_dataset)
    test_preds = torch.zeros(test_dataset_len).to(device=device)
    test_labels = torch.zeros(test_dataset_len).to(device=device)
    test_weights = torch.zeros(test_dataset_len).to(device=device)
    test_isnoise = torch.zeros(test_dataset_len).to(device=device)
    test_isnoise_labels = torch.zeros(test_dataset_len).to(device=device)
    last_step = 0
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            # get the date and move data to device and forward
            features = batch_data["feature"].to(device=device)  # (bs, 3, 8, 8)
            labels = batch_data["label"].to(device=device)  # (bs, 1)
            weights = batch_data["weight"].to(device=device)  # (bs, 1)
            isnoise_labels = batch_data["is_noise"].to(device=device)  # (bs, )
            outputs = model(features)
            preds, isnoise = outputs["output"], outputs["is_noise"]
            # note the result in one iter
            now_step = last_step + preds.shape[0]
            test_preds[last_step:now_step] = preds[:, 0].detach()
            test_labels[last_step:now_step] = labels[:, 0].detach()
            test_weights[last_step:now_step] = weights[:, 0].detach()
            test_isnoise[last_step:now_step] = isnoise.argmax(axis=1).detach()
            test_isnoise_labels[last_step:now_step] = isnoise_labels.detach()
            last_step = now_step

    # ---- Print result ---- #
    star_idx, end_idx = 0, 1
    r2 = r2_score(
        y_true=test_labels.cpu().numpy()[star_idx:end_idx], y_pred=test_preds.cpu().numpy()[star_idx:end_idx],
        weight=test_weights.cpu().numpy()[star_idx:end_idx]
    )
    acc = accuracy_score(y_true=test_isnoise_labels.cpu().numpy()[star_idx:end_idx], y_pred=test_isnoise.cpu().numpy()[star_idx:end_idx])
    f1 = f1_score(
        y_true=test_isnoise_labels.cpu().numpy()[star_idx:end_idx], y_pred=test_isnoise.cpu().numpy()[star_idx:end_idx],
        average="weighted"
    )
    print(f"r2 = {r2:.4f}")
    print(f"acc = {acc:.4f}")
    print(f"f1 = {f1:.4f}")
    logging.info("***************** TEST OVER ! *****************\n")


if __name__ == "__main__":
    pred_model()
