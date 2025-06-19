# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 10:41
# @Author  : Karry Ren

""" The dataset for Jane Street Market Prediction (JSMP).

Please follow the data preprocess, and get the following data structure:
├── Data/
    ├── dataset/
        ├── train.npz
        ├── valid.npz

ATTENTION:
    - Time Step is controlled by the data preprocess.

"""

from torch.utils import data
import numpy as np


class JSMPDataset(data.Dataset):
    """ The torch.Dataset of JSMP dataset. """

    def __init__(self, root_path: str, data_type: str = "train"):
        """ The init function of JSMPDataset.

        :param root_path: the root path of JSMP dataset
        :param data_type: the data_type of dataset, you have 3 choices now:
            - "train" for train dataset
            - "valid" for valid dataset

        """

        assert data_type in ["train", "valid"], "data_type should be 'train' or 'valid' !"

        # ---- Read the data and get x, y, w ---- #
        self.data = np.load(f"{root_path}/dataset_cnn/{data_type}.npz")
        self.feature, self.label, self.weight, self.is_noise = self.data["x"], self.data["y"], self.data["w"], self.data["n"]

        # ---- Check the length ---- #
        self.feature_len = self.feature.shape[0]
        self.label_len = self.label.shape[0]
        self.weight_len = self.weight.shape[0]
        self.is_noise_len = self.is_noise.shape[0]
        assert self.feature_len == self.label_len == self.weight_len == self.is_noise_len, "Length ERROR !"

        # ---- Set the data type ---- #
        self.feature = self.feature.reshape(self.feature_len, 3, 8, 8).astype("float32")
        self.label = self.label.astype("float32")
        self.weight = self.weight.astype("float32")
        self.is_noise = (self.is_noise.sum(axis=1) != 0).astype("long")

    def __len__(self):
        """ Get the length of dataset. """

        return self.feature_len

    def __getitem__(self, idx: int):
        """ Get the item based on idx.

        return: item_data
            - `feature`: the feature, shape=(time_steps, 8, 8)
            - `label`: the responder label, shape=(1, )
            - `weight`: the weight, shape=(1, )
            - `is_noise`: the is_noise label, shape=()

        """

        # ---- Construct item data ---- #
        item_data = {
            "feature": self.feature[idx],
            "label": self.label[idx],
            "weight": self.weight[idx],
            "is_noise": self.is_noise[idx]
        }

        return item_data


if __name__ == "__main__":  # a demo using JSMPDataset
    JSMP_DATASET_PATH = "../../Data/"
    data_set = JSMPDataset(JSMP_DATASET_PATH, data_type="train")
    for i in range(100):
        print(data_set[i]["feature"].max())
        print(data_set[i]["is_noise"])
        print(data_set[i]["feature"].shape, data_set[0]["label"].shape, data_set[0]["weight"].shape, data_set[0]["is_noise"].shape)
        print(type(data_set[i]["feature"][0, 0, 0]), type(data_set[i]["label"][0]), type(data_set[i]["weight"][0]), type(data_set[i]["is_noise"]))
