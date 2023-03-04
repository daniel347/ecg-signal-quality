import pandas as pd
import os
import scipy
import numpy as np
from DiagEnum import DiagEnum

# -------- Loading the datasets --------

cinc_pk_path = "../Datasets/CinC2017Data/database.pk"

training_path = "../Datasets/CinC2017Data/training2017/training2017/"
answers_path = "../Datasets/CinC2017Data/REFERENCE-v3.csv"


def load_cinc_dataset_scratch():
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

    dataset.to_pickle(cinc_pk_path)
    return dataset


def load_cinc_dataset_pickle():
    try:
        dataset = pd.read_pickle(cinc_pk_path)
        return dataset
    except (OSError, FileNotFoundError):
        return


def load_cinc_dataset(force_reload=False):
    dataset = None
    if not force_reload:
        dataset = load_cinc_dataset_pickle()
    if dataset is None:
        print("Failed to load from pickle, regenerating files")
        dataset = load_cinc_dataset_scratch()

    dataset = add_extra_columns(dataset)

    return dataset


def generate_safer_style_label(c):
    if c == "N":
        return DiagEnum.NoAF
    if c == "O":
        return DiagEnum.CannotExcludePathology
    if c == "A":
        return DiagEnum.AF
    if c == "~":
        return DiagEnum.PoorQuality


def add_extra_columns(dataset):
    dataset["length"] = dataset["data"].map(lambda arr: arr.shape[-1])
    dataset["data"] = dataset["data"].map(lambda d: d[0])

    # Normalise over the entire signal
    dataset["data"] = (dataset["data"] - dataset["data"].map(lambda x: x.mean())) / dataset["data"].map(
        lambda x: x.std())
    dataset["class"] = dataset["class"].map(generate_safer_style_label)
    dataset["class_index"] = (dataset["class"] == DiagEnum.PoorQuality).astype(int)
    return dataset


# -------- Preprocessing --------
def adaptive_gain_norm(x, w):
    x_mean_sub = np.pad(x - x.mean(), int((w - 1) / 2), "reflect")
    window = np.ones(w)
    sigma_square = np.convolve(x_mean_sub ** 2, window, mode="valid") / w
    gain = 1 / np.sqrt(sigma_square)

    return x * gain
