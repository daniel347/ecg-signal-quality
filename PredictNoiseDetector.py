import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from DataHandlers.DiagEnum import DiagEnum
from Models.NoiseCNN import CNN, hyperparameters
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from torch.fft import fft

# from DataHandlers.Dataloaders import ECGDataset as Dataset
# from DataHandlers.Dataloaders import DatasetSequenceIterator, load_feas1_chunk_range, prepare_safer_data
# from DataHandlers.SAFERDataset import feas1_path, feas2_path, num_chunks, chunk_size
from DataHandlers.SAFERDatasetV2 import SaferDataset, feas1_path, feas2_path, trial_path, num_chunks, chunk_size
from DataHandlers.CinC2020Dataset import cinc_2020_path
from DataHandlers.CinCDataset import cinc_2017_path
from DataHandlers.DataProcessUtilities import adaptive_gain_norm

from Utilities.Predict import *
from Utilities.General import get_torch_device
from Utilities.Plotting import plot_ecg


# ====  Options  ====
enable_cuda = True
model_name = "CNN_16_may_final_no_undecided" # "cnn_model_trial_trained_10_jun_2024_adaptive_gain" #


def filter_ecgs_with_split(pt, ecg, pt_split=None):
    ecg_new = ecg[ecg.length == 9120]
    ecg_new = ecg_new[ecg_new.measDiag != DiagEnum.Undecided]
    ecg_new = ecg_new[ecg_new.measDiagAgree |
                     (ecg_new.measDiagRev1 == DiagEnum.Undecided) |
                     (ecg_new.measDiagRev2 == DiagEnum.Undecided)]
    pt_new = pt[pt.ptID.isin(ecg_new.ptID)]

    if pt_split is not None:
        # Select only participants from pt_split
        pt_new = pt_new[pt_new.ptID.isin(pt_split.index)]
        ecg_new = ecg_new[ecg_new.ptID.isin(pt_new.index)]

    return pt_new, ecg_new


def label_noise(x):
    return int(x == DiagEnum.PoorQuality)


def preprocess(x):
    return adaptive_gain_norm(x, 601)


if __name__ == "__main__":
    # Either a string or a list of strings of dataset names
    # val_pts = pd.read_pickle(os.path.join(trial_path, f"splits/9_Jun_2024_trial_val_pts.pk"))
    val_pts = None
    dataset_name = SaferDataset(feas=1,
                                label_gen=label_noise,
                                filter_func=lambda pt, ecg: filter_ecgs_with_split(pt, ecg, val_pts))
                                # )  # ["18_Jun_feas1_test_train_pts", "18_Jun_feas1_test_test_pts", "18_Jun_feas1_test_val_pts"]
    # Modify to local dataset path if not SAFER data
    if isinstance(dataset_name, list):
        dataset_path = [os.path.join(feas1_path, f"ECGs/{name}.pk") for name in dataset_name]
        output_path = os.path.join(feas1_path, f"ECGs/{dataset_name[0]}_noise_predictions.pk")
    elif isinstance(dataset_name, str):
        dataset_path = os.path.join(feas1_path, f"ECGs/{dataset_name}.pk")
        output_path = os.path.join(feas1_path, f"ECGs/{dataset_name}_noise_predictions.pk")
    elif isinstance(dataset_name, Dataset):
        output_path = f"noise_predictions_{dataset_name.feas}_11_june.pk"



    data_is_feas1_pt = False  # True if dataset_split_name contains patients from safer
    batch_size = 128
    # =======

    device = get_torch_device(enable_cuda)

    if data_is_feas1_pt:
        if type(dataset_name) == list:
            datasets = []
            for name in dataset_path:
                datasets.append(pd.read_pickle(name))
            pt_dataset = pd.concat(datasets)
        else:
            pt_dataset = pd.read_pickle(dataset_path)
        def filter_pts(ecg_data):
            return ecg_data[ecg_data["ptID"].isin(pt_dataset["ptID"])]

        def load_feas1_nth_chuck(n):
            ecg_data, pt_data = load_feas1_chunk_range((n, n + 1))
            ecg_data.index = ecg_data["measID"]
            pt_data.index = pt_data["ptID"]
            return prepare_safer_data(pt_data, ecg_data)[1]


        loading_functions = [lambda n=n: load_feas1_nth_chuck(n) for n in
                             range(num_chunks)]

        dataloader = DatasetSequenceIterator(loading_functions,
                                             [batch_size for n in range(num_chunks)],
                                             filter=filter_pts,
                                             noise_detection=True)
    else:
        if isinstance(dataset_name, list):
            datasets = []
            for name in dataset_path:
                datasets.append(pd.read_pickle(name))
            dataset = pd.concat(datasets)
        elif isinstance(dataset_name, str):
            dataset = pd.read_pickle(dataset_path)


        if isinstance(dataset_name, Dataset):
            torch_dataset = dataset_name
            dataset = torch_dataset.ecg_data
            dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=2)
        else:
            dataset = dataset[dataset["length"] == 9120]
            torch_dataset = Dataset(dataset)
            dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    noiseDetector = CNN(**hyperparameters).to(device)
    noiseDetector.load_state_dict(torch.load(f"TrainedModels/{model_name}.pt", map_location=device))
    noiseDetector.eval()

    inds = []
    noise_ps = []

    with torch.no_grad():
        for i, (signals, labels, ind) in tqdm(enumerate(dataloader), total=len(dataloader)):
            """
            print(f"Signal mean: {signals.mean(dim=(-1,))}")
            print(f"Signal var: {signals.var(dim=(-1,))}")

            plot_ecg(signals[0], fs=300, n_split=3)
            plt.show()
            plt.plot(torch.abs(fft(signals[0]))[:300])
            plt.show()
            """

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



