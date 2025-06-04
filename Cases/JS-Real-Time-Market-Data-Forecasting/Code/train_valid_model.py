# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 11:17
# @Author  : Karry Ren

""" Train and validate models. """

import os
import logging
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from models.net import Multi_CNN
from models.loss import MSE_Loss, CE_Loss
from models.metrics import r2_score, accuracy_score, f1_score
from datasets.jsmp_dataset import JSMPDataset
import config as config
from utils import fix_random_seed


def train_valid_model() -> None:
    """ Train & Valid Model. """
    logging.info(f"***************** RUN TRAIN&VALID MODEL *****************")

    # ---- Get the train and valid device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** IN DEVICE `{device}` *****************")

    # ---- Make the dataset and dataloader ---- #
    # make the dataset and dataloader of training
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    train_dataset = JSMPDataset(root_path=config.JSMP_DATASET_PATH, data_type="train")
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)  # the train dataloader
    logging.info(f"**** VALID DATASET & DATALOADER ****")
    valid_dataset = JSMPDataset(root_path=config.JSMP_DATASET_PATH, data_type="valid")
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the valid dataloader
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Train dataset: length = {len(train_dataset)}")
    logging.info(f"Valid dataset: length = {len(valid_dataset)}")

    # ---- Construct the model and transfer device, while making loss and optimizer ---- #
    logging.info(f"***************** BEGIN BUILD UP THE MODEL ! *****************")
    model = Multi_CNN().to(device)
    mse_loss, ce_loss = MSE_Loss(), CE_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # ---- Start Train and Valid ---- #
    # init the metric dict of all epochs
    epoch_metric = {
        # train & valid loss
        "train_loss": np.zeros(config.EPOCHS), "valid_loss": np.zeros(config.EPOCHS),
        # train & valid R2
        "train_R2": np.zeros(config.EPOCHS), "valid_R2": np.zeros(config.EPOCHS),
        # train & valid is noise ACC
        "train_ACC": np.zeros(config.EPOCHS), "valid_ACC": np.zeros(config.EPOCHS),
        # train & valid is noise F1
        "train_F1": np.zeros(config.EPOCHS), "valid_F1": np.zeros(config.EPOCHS)
    }

    # train model epoch by epoch
    logging.info(f"***************** BEGIN TRAINING AND VALID THE MODEL ! *****************")
    # start train and valid during train
    for epoch in tqdm(range(config.EPOCHS)):
        # start timer for one epoch
        t_start = datetime.now()
        # set the array for one epoch to store (all empty)
        train_loss_one_epoch, valid_loss_one_epoch = [], []
        train_dataset_len, valid_dataset_len = len(train_dataset), len(valid_dataset)
        train_preds_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_labels_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_weights_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_isnoise_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_isnoise_labels_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        valid_preds_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_labels_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_weights_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_isnoise_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_isnoise_labels_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        # - TRAIN model
        last_step = 0
        model.train()
        for batch_data in tqdm(train_loader):
            # get the date and move data to device
            features = batch_data["feature"].to(device=device)  # (bs, 3, 8, 8)
            labels = batch_data["label"].to(device=device)  # (bs, 1)
            weights = batch_data["weight"].to(device=device)  # (bs, 1)
            isnoise_labels = batch_data["is_noise"].to(device=device)  # (bs, )
            # zero_grad, forward, compute loss, backward and optimize
            optimizer.zero_grad()
            outputs = model(features)
            preds, isnoise = outputs["output"], outputs["is_noise"]
            loss = mse_loss(y_true=labels, y_pred=preds, weight=weights) + ce_loss(y_true=isnoise_labels, y_pred=isnoise)
            loss.backward()
            optimizer.step()
            # note the loss of training in one iter
            train_loss_one_epoch.append(loss.item())
            # note the result in one iter
            now_step = last_step + preds.shape[0]
            train_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
            train_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
            train_weights_one_epoch[last_step:now_step] = weights[:, 0].detach()
            train_isnoise_one_epoch[last_step:now_step] = isnoise.argmax(axis=1).detach()
            train_isnoise_labels_one_epoch[last_step:now_step] = isnoise_labels.detach()
            last_step = now_step
        # -- note the loss and metrics for one epoch of TRAINING
        epoch_metric["train_loss"][epoch] = np.mean(train_loss_one_epoch)
        epoch_metric["train_R2"][epoch] = r2_score(
            y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy(), weight=train_weights_one_epoch.cpu().numpy()
        )
        epoch_metric["train_ACC"][epoch] = accuracy_score(
            y_true=train_isnoise_labels_one_epoch.cpu().numpy(), y_pred=train_isnoise_one_epoch.cpu().numpy()
        )
        epoch_metric["train_F1"][epoch] = f1_score(
            y_true=train_isnoise_labels_one_epoch.cpu().numpy(), y_pred=train_isnoise_one_epoch.cpu().numpy(),
            average="weighted"
        )

        # - VALID model
        last_step = 0
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(valid_loader):
                # get the date and move data to device and forward
                features = batch_data["feature"].to(device=device)  # (bs, 3, 8, 8)
                labels = batch_data["label"].to(device=device)  # (bs, 1)
                weights = batch_data["weight"].to(device=device)  # (bs, 1)
                isnoise_labels = batch_data["is_noise"].to(device=device)  # (bs, )
                outputs = model(features)
                preds, isnoise = outputs["output"], outputs["is_noise"]
                loss = mse_loss(y_true=labels, y_pred=preds, weight=weights) + ce_loss(y_true=isnoise_labels, y_pred=isnoise)
                # note the loss of validation in one iter
                valid_loss_one_epoch.append(loss.item())
                # note the result in one iter
                now_step = last_step + preds.shape[0]
                valid_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
                valid_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
                valid_weights_one_epoch[last_step:now_step] = weights[:, 0].detach()
                valid_isnoise_one_epoch[last_step:now_step] = isnoise.argmax(axis=1).detach()
                valid_isnoise_labels_one_epoch[last_step:now_step] = isnoise_labels.detach()
                last_step = now_step
            # -- note the loss and metrics for one epoch of VALIDATION
            epoch_metric["valid_loss"][epoch] = np.mean(valid_loss_one_epoch)
            epoch_metric["valid_R2"][epoch] = r2_score(
                y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy(), weight=valid_weights_one_epoch.cpu().numpy()
            )
            epoch_metric["valid_ACC"][epoch] = accuracy_score(
                y_true=valid_isnoise_labels_one_epoch.cpu().numpy(), y_pred=valid_isnoise_one_epoch.cpu().numpy()
            )
            epoch_metric["valid_F1"][epoch] = f1_score(
                y_true=valid_isnoise_labels_one_epoch.cpu().numpy(), y_pred=valid_isnoise_one_epoch.cpu().numpy(),
                average="weighted"
            )

        # - save model&model_config and metrics
        torch.save(model, f"{config.MODEL_SAVE_PATH}/model_pytorch_epoch_{epoch}")
        pd.DataFrame(epoch_metric).to_csv(f"{config.MODEL_SAVE_PATH}/model_metric.csv")

        # write metric log
        dt = datetime.now() - t_start
        logging.info(
            f"Epoch {epoch + 1}/{config.EPOCHS}, Duration: {dt}, {['%s:%.4f ' % (key, value[epoch]) for key, value in epoch_metric.items()]}"
        )

    # draw figure of train and valid metrics
    plt.figure(figsize=(15, 6))
    plt.subplot(3, 1, 1)
    plt.plot(epoch_metric["train_loss"], label="train loss", color="g")
    plt.plot(epoch_metric["valid_loss"], label="valid loss", color="b")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(epoch_metric["train_R2"], label="train R2", color="g")
    plt.plot(epoch_metric["valid_R2"], label="valid R2", color="b")
    plt.legend()
    plt.subplot(3, 2, 5)
    plt.plot(epoch_metric["train_ACC"], label="train ACC", color="g")
    plt.plot(epoch_metric["valid_ACC"], label="valid ACC", color="b")
    plt.legend()
    plt.subplot(3, 2, 6)
    plt.plot(epoch_metric["train_F1"], label="train F1", color="g")
    plt.plot(epoch_metric["valid_F1"], label="valid F1", color="b")
    plt.legend()
    plt.savefig(f"{config.SAVE_PATH}/training_steps.png", dpi=200, bbox_inches="tight")
    logging.info("***************** TRAINING OVER ! *****************\n")


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
