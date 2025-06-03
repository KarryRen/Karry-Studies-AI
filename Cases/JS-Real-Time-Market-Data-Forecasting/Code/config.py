# -*- coding: utf-8 -*-
# @Time    : 2025/6/1 21:53
# @Author  : Karry Ren

""" Some configs for operation. """

DATA_COLUMNS = [
    "date_id", "time_id", "symbol_id",
    "weight",
    "feature_00", "feature_01", "feature_02", "feature_03", "feature_04", "feature_05", "feature_06", "feature_07",
    "feature_08", "feature_09", "feature_10", "feature_11", "feature_12", "feature_13", "feature_14", "feature_15",
    "feature_16", "feature_17", "feature_18", "feature_19", "feature_20", "feature_21", "feature_22", "feature_23",
    "feature_24", "feature_25", "feature_26", "feature_27", "feature_28", "feature_29", "feature_30", "feature_31",
    "feature_32", "feature_33", "feature_34", "feature_35", "feature_36", "feature_37", "feature_38", "feature_39",
    "feature_40", "feature_41", "feature_42", "feature_43", "feature_44", "feature_45", "feature_46", "feature_47",
    "feature_48", "feature_49", "feature_50", "feature_51", "feature_52", "feature_53", "feature_54", "feature_55",
    "feature_56", "feature_57", "feature_58", "feature_59", "feature_60", "feature_61", "feature_62", "feature_63",
    "feature_64", "feature_65", "feature_66", "feature_67", "feature_68", "feature_69", "feature_70", "feature_71",
    "feature_72", "feature_73", "feature_74", "feature_75", "feature_76", "feature_77", "feature_78",
    "responder_6",
]

SKIP_DATES = 500

SELECTED_COLUMNS = [
    "date_id", "time_id", "symbol_id",
    "weight",
    "feature_00", "feature_01", "feature_02", "feature_03", "feature_04", "feature_05", "feature_06", "feature_07",
    "feature_08", "feature_09", "feature_10", "feature_11", "feature_12", "feature_13", "feature_14", "feature_16",
    "feature_17", "feature_18", "feature_19", "feature_20", "feature_22", "feature_23", "feature_24", "feature_25",
    "feature_28", "feature_29", "feature_30", "feature_32", "feature_33", "feature_34", "feature_35", "feature_36",
    "feature_37", "feature_38", "feature_40", "feature_43", "feature_45", "feature_46", "feature_47", "feature_48",
    "feature_49", "feature_51", "feature_54", "feature_56", "feature_57", "feature_58", "feature_59", "feature_60",
    "feature_61", "feature_62", "feature_63", "feature_64", "feature_65", "feature_66", "feature_67", "feature_68",
    "feature_69", "feature_70", "feature_71", "feature_72", "feature_75", "feature_76", "feature_77", "feature_78",
    "responder_6",
]

RANDOM_SEED = 42
SAVE_PATH = "./save/"
MODEL_SAVE_PATH = f"{SAVE_PATH}/models"
LOG_FILE = f"{SAVE_PATH}/log.log"
NUM_OF_VALID_DATES = 20
TIME_STEPS = 3
JSMP_DATASET_PATH = "../Data/"
BATCH_SIZE = 2048
LR = 1e-4
EPOCHS = 10
