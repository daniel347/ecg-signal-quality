import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy
import numpy as np
from enum import Enum
import wfdb
from DiagEnum import DiagEnum, feas1DiagToEnum

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


def load_feas_dataset_scratch(process, feas, ecg_range, ecg_meas_diag, save_name):
    dataset_path = feas2_path if (feas == 2) else feas1_path

    pt_data = pd.read_csv(os.path.join(dataset_path, "pt_data_anon.csv"))
    ecg_data = pd.read_csv(os.path.join(dataset_path, "rec_data_anon.csv"))
    ecg_data["data"] = None
    ecg_data["adc_gain"] = None

    ecg_data["measDiagAgree"] = ecg_data["measDiagRev1"] == ecg_data["measDiagRev2"]

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
    ecg_data["class_index"] = (ecg_data["measDiag"] == DiagEnum.PoorQuality).astype(int)
    ecg_data["length"] = ecg_data["data"].map(lambda x: x.shape[-1])
    ecg_data.to_pickle(os.path.join(dataset_path, f"ECGs/raw_{save_name}.pk"))

    if process:
        ecg_data = process_data(ecg_data)
        ecg_data.to_pickle(os.path.join(dataset_path, f"ECGs/filtered_{save_name}.pk"))

    return pt_data, ecg_data


def process_data(feas2_ecg_data, f_low=0.67, f_high=30, resample_rate=60):
    # perform band pass filtering, notch filtering (for power line interference)
    sos = scipy.signal.butter(3, [f_low, f_high], 'bandpass', fs=500, output='sos')
    sos_notch = scipy.signal.butter(3, [48, 52], 'bandstop', fs=500, output='sos')

    def filter_and_norm(x, sos):
        x_filt = scipy.signal.sosfiltfilt(sos, x, padlen=150)
        x_norm = (x_filt - x_filt.mean()) / x_filt.std()
        return x_norm

    """
    w, h = scipy.signal.sosfreqz(sos, fs=500 * 2 * np.pi)
    plt.semilogx(w/(2*np.pi), 20 * np.log10(np.abs(h)))
    plt.title('band pass frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.show()

    w, h = scipy.signal.sosfreqz(sos_notch, fs=500 * 2 * np.pi)
    plt.semilogx(w/(2*np.pi), 20 * np.log10(np.abs(h)))
    plt.title('notch frequency response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.show()
    """

    feas2_ecg_data["data"] = feas2_ecg_data["data"].map(lambda x: filter_and_norm(x, sos))
    feas2_ecg_data["data"] = feas2_ecg_data["data"].map(lambda x: filter_and_norm(x, sos_notch))

    if resample_rate != 500:
        def resample(x):
            resample_len = int(round(x.shape[-1] * resample_rate/500))
            return scipy.signal.resample(x, resample_len)

        feas2_ecg_data["data"] = feas2_ecg_data["data"].map(lambda x: resample(x))
        feas2_ecg_data["length"] = feas2_ecg_data["data"].map(lambda x: x.shape[-1])

    return feas2_ecg_data


def load_feas_dataset_pickle(process, f_name, feas=2, force_reprocess=False):
    dataset_path = feas2_path if (feas == 2) else feas1_path

    try:
        pt_data = pd.read_csv(os.path.join(dataset_path, "pt_data_anon.csv"))
        end_path = r"ECGs\filtered_dataframe.pk" if (process and not force_reprocess) else f"ECGs/raw_{f_name}.pk"
        ecg_data = pd.read_pickle(os.path.join(dataset_path, end_path))

        if force_reprocess:
            ecg_data = process_data(ecg_data)
            ecg_data.to_pickle(os.path.join(dataset_path, f"ECGs/filtered_{f_name}.pk"))

        return pt_data, ecg_data
    except (OSError, FileNotFoundError, Exception):
        return


def load_feas_dataset(feas=2, save_name="dataframe.pk", force_reload=False, process=True, force_reprocess=False, ecg_range=None, ecg_meas_diag=None):
    dataset = None
    if not force_reload:
        dataset = load_feas_dataset_pickle(process, save_name, feas, force_reprocess)
    if dataset is None:
        print("Failed to load from pickle, regenerating files")
        dataset = load_feas_dataset_scratch(process, feas, ecg_range, ecg_meas_diag, save_name=save_name)
    else:
        dataset = filter_dataset(dataset[0], dataset[1], ecg_range, ecg_meas_diag)

    return dataset
