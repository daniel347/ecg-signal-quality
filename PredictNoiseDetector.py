import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from Models.NoiseCNN import CNN, hyperparameters

from DataHandlers.Dataloaders import ECGDataset as Dataset
from DataHandlers.Dataloaders import DatasetSequenceIterator, load_feas1_chunk_range, prepare_safer_data

import Utilities.constants as constants
from Utilities.Predict import *
from Utilities.General import get_torch_device


# ====  Options  ====
enable_cuda = True
model_name = "20_Jun_noise_detector_test_script"

dataset_name = "19_Jun_from_scratch_test"
# Modify to local dataset store if not SAFER data
dataset_path = os.path.join(constants.feas2_path, f"ECGs/{dataset_name}.pk")
output_path = os.path.join(constants.feas2_path, f"ECGs/{dataset_name}_noise_predictions.pk")

data_is_feas1_pt = False  # True if dataset_split_name contains patients from safer
batch_size = 128
# =======

device = get_torch_device(enable_cuda)

if data_is_feas1_pt:
    pt_dataset = pd.read_pickle(dataset_path)
    def filter_pts(ecg_data):
        return ecg_data[ecg_data["ptID"].isin(pt_dataset["ptID"])]

    def load_feas1_nth_chuck(n):
        ecg_data, pt_data = load_feas1_chunk_range((n, n + 1))
        ecg_data.index = ecg_data["measID"]
        pt_data.index = pt_data["ptID"]
        return prepare_safer_data(pt_data, ecg_data)[1]


    loading_functions = [lambda n=n: load_feas1_nth_chuck(n) for n in
                         range(constants.num_chunks)]

    dataloader = DatasetSequenceIterator(loading_functions,
                                         [batch_size for n in range(constants.num_chunks)],
                                         filter=filter_pts,
                                         noise_detection=True)
else:
    dataset = pd.read_pickle(dataset_path)
    dataset = dataset[dataset["length"] == 9120]
    torch_dataset = Dataset(dataset)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

noiseDetector = CNN(**hyperparameters).to(device)
noiseDetector.load_state_dict(torch.load(f"TrainedModels/{model_name}.pt", map_location=device))
noiseDetector.eval()

inds = []
noise_ps = []

with torch.no_grad():
    for i, (signals, labels, ind) in enumerate(dataloader):
        signal = signals.to(device).float()
        noise_prob = noiseDetector(torch.unsqueeze(signal, 1)).detach().to("cpu").numpy()

        for i, n in zip(ind, noise_prob):
            if type(i) == str:
                inds.append(i)
            else:
                inds.append(i.item())
            noise_ps.append(float(n))

noise_predictions = pd.Series(data=noise_ps, index=inds)
noise_predictions.to_pickle(output_path)

if not data_is_feas1_pt:
    dataset["noise_prediction"] = noise_predictions

conf_mat = confusion_matrix(dataset["class_index"], dataset["noise_prediction"] > 0)
print_noise_results(conf_mat)



