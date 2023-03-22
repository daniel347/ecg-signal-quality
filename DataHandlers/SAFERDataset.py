import pandas as pd
import os
import scipy
import wfdb
import numpy as np
from DataHandlers.DiagEnum import DiagEnum, feas1DiagToEnum
from DataHandlers.DataProcessUtilities import *
from multiprocesspandas import applyparallel

import matplotlib
matplotlib.use('TkAgg')

# -------- Loading the datasets --------

feas2_path = r"D:\2022_23_DSiromani\Feas2"
feas1_path = r"D:\2022_23_DSiromani\Feas1"


def filter_dataset(pt_data, ecg_data, ecg_range, ecg_meas_diag):
    if ecg_range is not None:
        # We read only some of the ecgs - particularly for feas1 this breaks data into manageable chunks
        ecg_data = ecg_data[(ecg_data["measID"] >= ecg_range[0]) & (ecg_data["measID"] < ecg_range[1])]
        pt_data = pt_data[pt_data["ptID"].isin(ecg_data["ptID"])]

    if ecg_meas_diag is not None:
        ecg_data = ecg_data[ecg_data["measDiag"].isin(ecg_meas_diag)]
        pt_data = pt_data[pt_data["ptID"].isin(ecg_data["ptID"])]

    return pt_data, ecg_data


def generate_af_class_labels(dataset):
    """See notes/ emails for explaination of logic"""
    dataset["class_index"] = -1
    # If not tagged assume Normal
    dataset.loc[(dataset["not_tagged_ign_wide_qrs"] == 1) & (dataset["measDiag"] == DiagEnum.Undecided), "class_index"] = 0

    # The first review rejection is Normal (more or less)
    dataset.loc[(dataset["not_tagged_ign_wide_qrs"] == 0) & (dataset["ffReview_sent"] == 1) & (dataset["ffReview_remain"] == 0) & (dataset["feas"] == 1), "class_index"] = 0

    # dataset.loc[(dataset["not_tagged_ign_wide_qrs"] == 0) & (dataset["feas"] == 2), "class_index"] = 2 # Anything tagged in feas 2 goes to other

    dataset.loc[dataset["measDiag"] == DiagEnum.AF, "class_index"] = 1  # The cardiologist has said AF

    # Anything thats got this far is probably dodgy in some way
    dataset.loc[dataset["measDiag"].isin([DiagEnum.NoAF, DiagEnum.HeartBlock]), "class_index"] = 2

    dataset.loc[(dataset["measDiag"].isin([DiagEnum.CannotExcludePathology, DiagEnum.PoorQuality])), "class_index"] = -1

    dataset["measDiagAgree"] = (dataset["measDiagRev1"] == dataset["measDiagRev2"]) |\
                               (dataset["measDiagRev1"] == DiagEnum.Undecided) |\
                               (dataset["measDiagRev2"] == DiagEnum.Undecided)
    dataset.loc[~dataset["measDiagAgree"], "class_index"] = -1

    dataset = dataset[dataset["class_index"] != -1]

    return dataset


def add_ecg_class_counts(safer_pt_data, safer_ecg_data):
    safer_pt_data["noNormalRecs"] = safer_ecg_data[safer_ecg_data["class_index"] == 0]["ptID"].value_counts()
    safer_pt_data["noAFRecs"] = safer_ecg_data[safer_ecg_data["class_index"] == 1]["ptID"].value_counts()
    safer_pt_data["noOtherRecs"] = safer_ecg_data[safer_ecg_data["class_index"] == 2]["ptID"].value_counts()

    safer_pt_data["noAFRecs"] = safer_pt_data["noAFRecs"].fillna(0)
    safer_pt_data["noNormalRecs"] = safer_pt_data["noNormalRecs"].fillna(0)
    safer_pt_data["noOtherRecs"] = safer_pt_data["noOtherRecs"].fillna(0)

    return safer_pt_data


def load_feas_dataset_scratch(process, feas, ecg_range, ecg_meas_diag, save_name, filter_func):
    dataset_path = feas2_path if (feas == 2) else feas1_path

    pt_data = load_pt_dataset(feas)
    ecg_data = load_ecg_csv(feas, pt_data, ecg_range, ecg_meas_diag, 10000, 200000)

    if filter_func:
        pt_data, ecg_data = filter_func(pt_data, ecg_data)

    # Read wfdb ECG records
    for ind, file_path in ecg_data["file_path"].iteritems():
        print(f"Reading file {file_path}\r", end="")
        try:
            record = wfdb.rdrecord(os.path.join(dataset_path, file_path))
            ecg_data["data"].loc[ind] = record.p_signal[:, 0]
            ecg_data["adc_gain"].loc[ind] = record.adc_gain[0]
        except (OSError, FileNotFoundError):
            print("Error, file does not exist or cannot be read, skipping")

    ecg_data.dropna(subset=["data"], inplace=True)

    # Generate the class_index
    ecg_data["length"] = ecg_data["data"].map(lambda x: x.shape[-1])
    ecg_data.to_pickle(os.path.join(dataset_path, f"ECGs/raw_{save_name}.pk"))

    if process:
        ecg_data = process_data(ecg_data)
        ecg_data.to_pickle(os.path.join(dataset_path, f"ECGs/filtered_{save_name}.pk"))

    return pt_data, ecg_data


def process_data(feas2_ecg_data, f_low=0.67, f_high=30, resample_rate=300):
    # perform band pass filtering, notch filtering (for power line interference)
    sos = scipy.signal.butter(3, [f_low, f_high], 'bandpass', fs=500, output='sos')
    sos_notch = scipy.signal.butter(3, [48, 52], 'bandstop', fs=500, output='sos')

    feas2_ecg_data["data"] = feas2_ecg_data["data"].apply_parallel(lambda x: filter_and_norm(x, sos))
    feas2_ecg_data["data"] = feas2_ecg_data["data"].apply_parallel(lambda x: filter_and_norm(x, sos_notch))

    if resample_rate != 500:
        feas2_ecg_data["data"] = feas2_ecg_data["data"].apply_parallel(lambda x: resample(x, resample_rate, 500))
        feas2_ecg_data["length"] = feas2_ecg_data["data"].map(lambda x: x.shape[-1])

    # Get beat positions and heartrate
    feas2_ecg_data["r_peaks"] = feas2_ecg_data.apply(get_r_peaks, axis=1)
    """
    try:
        feas2_ecg_data["r_peaks"] = feas2_ecg_data.apply(get_r_peaks, axis=1)
    except:
        feas2_ecg_data["r_peaks"] = feas2_ecg_data.apply(get_r_peaks, axis=1)
    """

    feas2_ecg_data["heartrate"] = feas2_ecg_data.apply(lambda e: (len(e["r_peaks"]) / (e["length"] / 300)) * 60, axis=1)

    # Get the rri feature
    feas2_ecg_data["rri_feature"] = (feas2_ecg_data["r_peaks"] / resample_rate).map(lambda x: get_rri_feature(x, 60))
    fewer_5_beats = feas2_ecg_data["rri_feature"].map(lambda x: np.sum(x == 0) > 55)
    feas2_ecg_data = feas2_ecg_data[~fewer_5_beats]
    feas2_ecg_data["rri_len"] = feas2_ecg_data["rri_feature"].map(lambda x: x[x > 0].shape[-1])
    feas2_ecg_data["rri_feature"] = normalise_rri_feature(feas2_ecg_data["rri_feature"])

    return feas2_ecg_data


def load_pt_dataset(feas, feas2_offset=10000):
    dataset_path = feas2_path if (feas == 2) else feas1_path
    pt_df = pd.read_csv(os.path.join(dataset_path, "pt_data_anon.csv"))
    if feas == 2:
        pt_df["ptID"] += feas2_offset
        pt_df.index = pt_df["ptID"]
    pt_df["feas"] = feas
    return pt_df


def load_ecg_csv(feas, pt_data, ecg_range, ecg_meas_diag, feas2_offset=10000, feas2_ecg_offset=200000):
    dataset_path = feas2_path if (feas == 2) else feas1_path
    ecg_data = pd.read_csv(os.path.join(dataset_path, "rec_data_anon.csv"))
    if feas2_offset != 0 and feas == 2:
        ecg_data["ptID"] += feas2_offset
    ecg_data["data"] = None
    ecg_data["adc_gain"] = None

    ecg_data["measDiagAgree"] = (ecg_data["measDiagRev1"] == ecg_data["measDiagRev2"]) | \
                                (ecg_data["measDiagRev1"] == DiagEnum.Undecided) | \
                                (ecg_data["measDiagRev2"] == DiagEnum.Undecided)

    # convert data to the Enum
    diag_columns = ["ptDiag", "ptDiagRev1", "ptDiagRev2", "ptDiagRev3", "measDiag", "measDiagRev1", "measDiagRev2"]
    for diag_ind in diag_columns:
        if feas == 2:
            ecg_data[diag_ind] = ecg_data[diag_ind].map(lambda d: DiagEnum(d))
        elif feas == 1:
            ecg_data[diag_ind] = ecg_data[diag_ind].map(lambda d: feas1DiagToEnum(d))

    pt_data, ecg_data = filter_dataset(pt_data, ecg_data, ecg_range, ecg_meas_diag)

    # Generate ECG file names
    ecg_path_labels = "ECGs/{:06d}/saferF{}_{:06d}"
    ecg_data["file_path"] = ecg_data["measID"].map(lambda id: ecg_path_labels.format((id // 1000) * 1000, feas, id))

    if feas2_ecg_offset != 0 and feas == 2:
        ecg_data["measID"] += feas2_ecg_offset

    # Generate class index
    ecg_data["class_index"] = (ecg_data["measDiag"] == DiagEnum.PoorQuality).astype(int)
    ecg_data["length"] = None
    ecg_data.index = ecg_data["measID"]

    return ecg_data


def load_feas_dataset_pickle(process, f_name, feas=2, force_reprocess=False):
    dataset_path = feas2_path if (feas == 2) else feas1_path

    try:
        pt_data = load_pt_dataset(feas, 10000)
        end_path = f"ECGs/filtered_{f_name}.pk" if (process and not force_reprocess) else f"ECGs/raw_{f_name}.pk"

        ecg_data = pd.read_pickle(os.path.join(dataset_path, end_path))

        if force_reprocess:
            ecg_data = process_data(ecg_data)
            ecg_data.to_pickle(os.path.join(dataset_path, f"ECGs/filtered_{f_name}.pk"))

        return pt_data, ecg_data
    except (OSError, FileNotFoundError) as e:
        print(e)
        return


def load_feas_dataset(feas=2, save_name="dataframe.pk", force_reload=False, process=True, force_reprocess=False, ecg_range=None, ecg_meas_diag=None, filter_func=None):
    dataset = None
    if not force_reload:
        dataset = load_feas_dataset_pickle(process, save_name, feas, force_reprocess)
    if dataset is None:
        print("Failed to load from pickle, regenerating files")
        dataset = load_feas_dataset_scratch(process, feas, ecg_range, ecg_meas_diag, save_name,  filter_func)
    else:
        dataset = filter_dataset(dataset[0], dataset[1], ecg_range, ecg_meas_diag)

    return dataset
