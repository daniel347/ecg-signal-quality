import sys
import pandas as pd
import os
import torch

from DataHandlers.DiagEnum import DiagEnum
import DataHandlers.SAFERDataset as SAFERDataset

from Models.NoiseCNN import CNN

from Utilities.Training import *
from DataHandlers.Dataloaders import RRiECGDataset as Dataset

import Utilities.constants as constants
import torch

# A fudge because I moved the files
sys.modules["DiagEnum"] = DiagEnum
sys.modules["SAFERDataset"] = SAFERDataset


# ====  Options  ====
enable_cuda = True
out_model_name = "Transformer_16_June_safer_train_attention_pooling_augmentation_smoothing"

pre_trained_model_name = "Transformer_16_June_cinc_2020_train_attention_pooling_augmentation_smoothing"
dataset_split_name = "feas1_27_mar"
dataset_type = "safer_feas1"

augment_noise = False
remove_noisy_samples = False

batch_size = 64
