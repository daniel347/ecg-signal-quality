import torch
import DataAugmentations
from DataProcessUtilities import *

import threading

from torch.utils.data import DataLoader
import pandas as pd
import sys

import DataHandlers.SAFERDataset as SAFERDataset
from DataHandlers.DiagEnum import DiagEnum
import Utilities.constants as constants

from torch.utils.data import Dataset


sys.modules["SAFERDataset"] = SAFERDataset
import math


class TransformerDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset):
        'Initialization'
        self.dataset = dataset
        self.noise_prob = 0
        self.temp_warp = 0

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset.index)

    def set_noise_prob(self, prob, power_std, noise_df):
        self.noise_prob = prob
        self.noise_power_std = power_std
        self.noise_df = noise_df

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.dataset.iloc[index]

        data = row["data"]
        rri = row["rri_feature"]
        rri_len = row["rri_len"]

        warp = np.random.binomial(1, self.temp_warp)
        if warp:
            data, r_peaks = DataAugmentations.temporal_warp(data, row["r_peaks_hamilton"])
            rri = get_rri_feature(r_peaks, 20)

        add_noise = np.random.binomial(1, self.noise_prob)
        if add_noise:
            noise = self.noise_df.sample()["data"].iloc[0] * np.random.normal(scale=self.noise_power_std)
            data += noise

        X = (data, rri, rri_len)
        y = row["class_index"]
        ind = row.name

        return X, y, ind


class DatasetSequenceIterator:

    def __init__(self, data_loading_functions, batch_sizes, filter=lambda x:x):
        self.dl_functions = data_loading_functions

        self.dataset = None
        self.next_dataset = None

        self.dataloader_iterator = None
        self.next_dataloader_iterator = None

        self.next_dataset_loaded = False
        self.dataloader_thread = None

        self.filter = filter

        self.batch_sizes = batch_sizes
        self.dl_index = 0

    def __iter__(self):
        self.dl_index = -1
        self.dataloader_thread = threading.Thread(target=self.load_next_dataset)
        self.dataloader_thread.start()
        self.dataloader_thread.join()
        self.swap_to_next_dataset()
        self.dl_index += 1
        self.dataloader_thread = threading.Thread(target=self.load_next_dataset)
        self.dataloader_thread.start()
        print(self.dl_index)
        return self

    def __len__(self):
        # TODO make this return the right value
        return 100

    def swap_to_next_dataset(self):
        self.dataset = self.next_dataset
        self.dataloader_iterator = self.next_dataloader_iterator
        self.next_dataset_loaded = False

    def load_next_dataset(self):
        if self.dl_index + 1 < len(self.dl_functions):
            print(f"Loading dataset {self.dl_index + 1}")
            self.next_dataset = self.dl_functions[self.dl_index + 1]()
            self.next_dataset = self.filter(self.next_dataset)

            torch_dataset = Dataset(self.next_dataset)
            self.next_dataloader_iterator = iter(DataLoader(torch_dataset, batch_size=self.batch_sizes[self.dl_index], shuffle=True, pin_memory=True))
            self.next_dataset_loaded = True
        else:
            print("Finished loading all datasets")
            self.next_dataset_loaded = False
            return None

    def __next__(self):
        try:
            ret = next(self.dataloader_iterator)
        except StopIteration:
            print("stop_iteration")
            if self.dl_index >= len(self.dl_functions):
                # We have gone through all the datasets
                print("Completed all datasets")
                raise StopIteration
            else:

                if not self.next_dataset_loaded:
                    print("waiting_for_next_dataset")
                    self.dataloader_thread.join()

                self.swap_to_next_dataset()
                self.dl_index += 1
                self.dataloader_thread = threading.Thread(target=self.load_next_dataset)
                self.dataloader_thread.start()
                ret = next(self.dataloader_iterator)

        return ret


def load_feas1_chunk_range(chunk_range=(0, constants.num_chunks)):
    ecg_data = []
    pt_data = []

    for chunk_num in range(chunk_range[0], chunk_range[1]):
        feas1_pt_data, feas1_ecg_data = SAFERDataset.load_feas_dataset(1, f"dataframe_{chunk_num}.pk")

        ecg_data.append(feas1_ecg_data)
        pt_data.append(feas1_pt_data)

    feas1_ecg_data = pd.concat(ecg_data)
    feas1_ecg_data["feas"] = 1
    feas1_ecg_data["rri_len"] = feas1_ecg_data["rri_feature"].map(lambda x: x[x > 0].shape[-1])
    feas1_pt_data = pd.concat(pt_data).drop_duplicates()

    return feas1_ecg_data, feas1_pt_data


def prepare_safer_data(pt_data, ecg_data):
    if "length" in ecg_data:
        ecg_data = ecg_data[ecg_data["length"] == 9120]

    ecg_data = ecg_data[ecg_data["measDiag"] != DiagEnum.PoorQuality]
    # ecg_data = ecg_data[ecg_data["tag_orig_Poor_Quality"] == 0]

    ecg_data = ecg_data[ecg_data["rri_len"] > 5]

    pt_data.index = pt_data["ptID"]
    ecg_data = SAFERDataset.generate_af_class_labels(ecg_data)
    pt_data = SAFERDataset.add_ecg_class_counts(pt_data, ecg_data)

    return pt_data, ecg_data