import pandas as pd
import os
import scipy

from .DiagEnum import DiagEnum
from DataHandlers.DataProcessUtilities import *

cinc_2017_path = "Datasets/CinC2017Data/"
training_path = os.path.join(cinc_2017_path, "training2017/training2017/")
answers_path = os.path.join(cinc_2017_path, "REFERENCE-v3.csv")


def load_cinc_dataset_scratch(save_name="database"):
    dataset = pd.read_csv(answers_path, header=None, names=["class"], index_col=0)
    dataset["data"] = None

    print(dataset.head())

    for root, dirs, files in os.walk(training_path):
        for name in files:
            try:
                name, ext = name.split(".")
            except ValueError:
                print("error, skipping file")
                continue
            if ext == "mat":
                mat_data = scipy.io.loadmat(os.path.join(root, name + "." + ext))
                dataset.loc[name]["data"] = mat_data["val"]
                print(f"Adding {name}\r", end="")

    dataset.to_pickle(os.path.join(cinc_2017_path, f"{save_name}.pk"))
    return dataset


def load_cinc_dataset_pickle(save_name="database"):
    try:
        dataset = pd.read_pickle(os.path.join(cinc_2017_path, f"{save_name}.pk"))
        return dataset
    except (OSError, FileNotFoundError) as e:
        print(e)
        return


def load_cinc_dataset(force_reload=False):
    dataset = None
    if not force_reload:
        dataset = load_cinc_dataset_pickle()
    if dataset is None:
        print("Failed to load from pickle, regenerating files")
        dataset = load_cinc_dataset_scratch()

    print("Raw data loaded")

    dataset["length"] = dataset["data"].map(lambda arr: arr.shape[-1])
    dataset["data"] = dataset["data"].map(lambda d: d[0])
    dataset["fs"] = 300

    print("Length an fs added")

    dataset["measDiag"] = dataset["class"].map(generate_safer_style_label)
    dataset["class_index"] = (dataset["class"] == DiagEnum.PoorQuality).astype(int)

    print("MeasDiag and class index added")

    dataset = process_data(dataset)

    return dataset


def process_data(ecg_data, f_low=0.67, f_high=30, resample_rate=300):
    # perform band pass filtering, notch filtering (for power line interference)
    sos = scipy.signal.butter(3, [f_low, f_high], 'bandpass', fs=500, output='sos')
    sos_notch = scipy.signal.butter(3, [48, 52], 'bandstop', fs=500, output='sos')

    ecg_data["data"] = ecg_data["data"].map(lambda x: filter_and_norm(x, sos))
    ecg_data["data"] = ecg_data["data"].map(lambda x: filter_and_norm(x, sos_notch))

    print("Filtered")

    if resample_rate != 300:
        ecg_data["data"] = ecg_data["data"].map(lambda x: resample(x, resample_rate, 300))
        ecg_data["length"] = ecg_data["data"].map(lambda x: x.shape[-1])

    print("Resampled")

    # Get beat positions and heartrate
    print("Detecting R-peaks")
    ecg_data["r_peaks"] = ecg_data.apply(get_r_peaks, axis=1)
    ecg_data["heartrate"] = ecg_data.apply(lambda e: (len(e["r_peaks"]) / (e["length"] / e["fs"])) * 60, axis=1)

    # Get the rri feature
    ecg_data["rri_feature"] = (ecg_data["r_peaks"] / resample_rate).map(lambda x: get_rri_feature(x, 60))
    fewer_5_beats = ecg_data["rri_feature"].map(lambda x: np.sum(x == 0) > 55)
    ecg_data = ecg_data[~fewer_5_beats]
    ecg_data["rri_len"] = ecg_data["rri_feature"].map(lambda x: x[x > 0].shape[-1])
    # ecg_data["rri_feature"] = normalise_rri_feature(ecg_data)

    return ecg_data


def generate_safer_style_label(c):
    if c == "N":
        return DiagEnum.NoAF
    if c == "O":
        return DiagEnum.CannotExcludePathology
    if c == "A":
        return DiagEnum.AF
    if c == "~":
        return DiagEnum.PoorQuality

