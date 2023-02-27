import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy
import numpy as np

import wfdb
from scipy.io import loadmat
from DiagEnum import DiagEnum, feas1DiagToEnum
from CinC2020Enums import Sex
from tqdm import tqdm
from ecgdetectors import Detectors

import matplotlib
matplotlib.use('TkAgg')

dataset_path = r"C:\Users\daniel\Documents\CambridgeSoftwareProjects\ecg-signal-quality\CinC2020Data"


class CinC2020DiagMapper:

    def __init__(self):
        self.diag_desc = pd.read_csv(os.path.join(dataset_path, r"training\dx_mapping_scored.csv"))
        self.diag_desc["SNOMED CT Code"] = self.diag_desc["SNOMED CT Code"].astype(int)

        self.identical_diags = {713427006: 59118001,
                                63593006: 284470004,
                                17338001: 427172004}

    def mapToDesc(self, num):
        try:
            desc = self.diag_desc.loc[num]["Dx"]
        except KeyError:
            return ""
        return desc

    def mapToSaferDiag(self, num):
        try:
            diag = DiagEnum(self.diag_desc.loc[num]["MeasDiag"])
        except KeyError:
            return DiagEnum.Undecided
        return diag

    def mapFromDiagCode(self, code):
        if code in self.identical_diags.keys():
            code = self.identical_diags[code]
        try:
            num = self.diag_desc[self.diag_desc["SNOMED CT Code"] == code].iloc[0].name
        except (KeyError, IndexError):
            return -1
        return num


def load_challenge_data_lead_1(filename):
    # First load the data
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    # Then the header info
    header_data = wfdb.rdheader(filename[:-4], pn_dir=None, rd_segments=False)

    # Now extract to series
    lead_1_ind = 0  # np.where(header_data.sig_name == "I")[0]
    data = data[lead_1_ind, :]

    age = -1
    sex = Sex.Undefined
    diagnosis_num = [-1]

    length = data.shape[0]

    for comment in header_data.comments:
        if "Age" in comment:
            try:
                age = int(comment.split(":")[-1].strip())
            except (ValueError, TypeError):
                pass
        elif "Sex" in comment:
            try:
                sex = Sex[comment.split(":")[-1].strip()]
            except (ValueError, TypeError, KeyError):
                pass
        elif "Dx" in comment:
            try:
                diagnosis_num = [int(d) for d in comment.split(":")[-1].strip().split(",")]
            except (ValueError, TypeError):
                pass

    if "st_petersburg_incart" in filename:
        split_data, diagnoses = split_st_petersburg_file(filename[:-4], data, header_data.fs * 10)
        return [pd.Series({"data": dat,
                             "fs": header_data.fs,
                             "adc_gain": header_data.adc_gain[lead_1_ind],
                             "age": age,
                             "sex": sex,
                             "diag_num": diag,
                             "overall_diags": diagnosis_num,
                             "length": header_data.fs * 10,
                             "filepath": filename[:-4]}) for dat, diag in zip(split_data, diagnoses)]

    data_series = pd.Series({"data": data,
                             "fs": header_data.fs,
                             "adc_gain": header_data.adc_gain[lead_1_ind],
                             "age": age,
                             "sex": sex,
                             "diag_num": diagnosis_num,
                             "length": length,
                             "filepath": filename[:-4]})

    return [data_series]


def safer_diag_to_class_list(ecg_data):
    def map_diag(x):
        if x == DiagEnum.AF:
            return 1
        elif x == DiagEnum.CannotExcludePathology or x == DiagEnum.HeartBlock:
            # Other arrythmia
            return 2
        else:
            # Normal ECG
            return 0

    ecg_data["class_index"] = ecg_data["measDiag"].map(map_diag)
    return ecg_data


def chal_diag_to_safer_diag(ecg_data, mapper):

    def map_diag(x):
        diag_list = list(map(mapper.mapToSaferDiag, x))
        # priority of diagnoses
        if DiagEnum.AF in diag_list:
            return DiagEnum.AF
        elif DiagEnum.HeartBlock in diag_list:
            return DiagEnum.HeartBlock
        elif DiagEnum.CannotExcludePathology in diag_list:
            return DiagEnum.CannotExcludePathology
        elif DiagEnum.NoAF in diag_list:
            return DiagEnum.NoAF
        else:
            return DiagEnum.Undecided

    ecg_data["measDiag"] = ecg_data["chal_diag_num"].map(map_diag)
    return ecg_data


def keep_challenge_diagnoses(data, mapper):
    # Filters a dataframe to only the CinC 2020 challenge diagnoses
    diag_list = mapper.diag_desc["SNOMED CT Code"].values

    def map_diag(d):
        challenge_diag = []
        for n in d:
            if n in diag_list:
                mapped_diag = mapper.mapFromDiagCode(n)
                if mapped_diag != -1:
                    challenge_diag.append(mapped_diag)
        return challenge_diag

    data["chal_diag_num"] = data["diag_num"].map(map_diag)
    return data[data["chal_diag_num"].str.len() != 0]


def split_st_petersburg_file(filename, data, split_len):
    ann = wfdb.rdann(filename, "atr")

    splits = np.arange(0, data.shape[0], split_len)

    data_splits = []
    diagnoses = []

    symbols = np.array(ann.symbol)
    afib = False

    # Note here the split array includes 0 so we get an empty array in data_splits[0],
    # which is filtered so the rest of the diags align with the data

    for i, (start, end) in enumerate(zip(splits, splits[1:])):
        discard = False
        diagnosis = []
        annotations = symbols[np.logical_and(ann.sample >= start, ann.sample < end)]
        if "+" in annotations:
            # Change of rhythm marker
            afib = not afib

        ann_vals, ann_counts = np.unique(annotations, return_counts=True)
        ann_dict = {v: c for v, c in zip(ann_vals, ann_counts)}
        total_beats = len(annotations)
        if afib:
            # Atrial fibrillation
            diagnosis.append(164889003)
        elif "N" in ann_dict.keys() and ann_dict["N"] == total_beats and not afib:
            # Sinus rhythm -  all normal beats and no AF
            diagnosis.append(426783006)

        if "V" in ann_dict.keys():
            # Add in PVC
            diagnosis.append(427172004)
        if "A" in ann_dict.keys():
            # Add in PAC
            diagnosis.append(84470004)
        if "R" in ann_dict.keys():
            # Add in RBBB
            diagnosis.append(59118001)
        if "S" in ann_dict.keys():
            # Add in SVPB
            diagnosis.append(284470004)

        for c in "nQFjB":
            if c in ann_dict.keys():
                # discard this segment
                discard = True
        if not discard:
            diagnoses.append(diagnosis)
            data_splits.append(data[start:end])

    return data_splits, diagnoses


def load_dataset_scratch(process, ecg_range, ecg_meas_diag, save_name):
    # Generate ECG file names
    header_files = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            g = os.path.join(root, f)
            if not f[0] == '.' and f[-3:] == 'mat' and os.path.isfile(g):
                header_files.append(g)

    num_files = len(header_files)
    series_list = []

    # Read ECG files and headers
    for i in tqdm(range(num_files)):
        data_series = load_challenge_data_lead_1(header_files[i])
        series_list.extend(data_series)

    ecg_data = pd.DataFrame(series_list)

    ecg_data = ecg_data[~ecg_data["data"].map(lambda x: np.any(np.isnan(x)))]
    ecg_data.dropna(subset=["data"], inplace=True)

    # keep only the 27 classes of data used in the challenge
    mapper = CinC2020DiagMapper()
    ecg_data = keep_challenge_diagnoses(ecg_data, mapper)
    ecg_data = chal_diag_to_safer_diag(ecg_data, mapper)
    ecg_data = safer_diag_to_class_list(ecg_data)

    ecg_data.to_pickle(os.path.join(dataset_path, f"training/raw_{save_name}.pk"))

    if process:
        ecg_data = process_data(ecg_data)
        ecg_data.to_pickle(os.path.join(dataset_path, f"training/filtered_{save_name}.pk"))

    return ecg_data


def process_data(ecg_data, f_low=0.67, f_high=30, resample_rate=300):
    # perform band pass filtering, notch filtering (for power line interference)

    fs_list = pd.unique(ecg_data["fs"])
    bandpass_dict = {fs: scipy.signal.butter(3, [f_low, f_high], 'bandpass', fs=fs, output='sos') for fs in fs_list}
    notch_dict = {fs: scipy.signal.butter(3, [48, 52], 'bandstop', fs=fs, output='sos') for fs in fs_list}

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

    ecg_data["data"] = ecg_data.apply(lambda x: filter_and_norm(x["data"], bandpass_dict[x["fs"]]), axis=1)
    ecg_data["data"] = ecg_data.apply(lambda x: filter_and_norm(x["data"], notch_dict[x["fs"]]), axis=1)

    def resample(x, orig_fs):
        resample_len = int(round(x.shape[-1] * resample_rate/orig_fs))
        return x if (x.shape[-1] == resample_len) else scipy.signal.resample(x, resample_len)

    ecg_data["data"] = ecg_data.apply(lambda x: resample(x["data"], x["fs"]), axis=1)
    ecg_data["length"] = ecg_data["data"].map(lambda x: x.shape[-1])
    ecg_data["fs"] = resample_rate

    # Get beat positions and heartrates
    detectors = Detectors(300)

    ecg_data["r_peaks"] = ecg_data["data"].map(detectors.pan_tompkins_detector)
    ecg_data["r_peaks"] = ecg_data["r_peaks"].map(np.array)
    ecg_data["heartrate"] = ecg_data.apply(lambda e: (len(e["r_peaks"]) / (e["length"] / e["fs"])) * 60, axis=1)

    return ecg_data


def load_dataset_pickle(process, f_name, force_reprocess=False):
    try:
        end_path = f"training/filtered_{f_name}.pk" if (process and not force_reprocess) else f"ECGs/raw_{f_name}.pk"
        ecg_data = pd.read_pickle(os.path.join(dataset_path, end_path))

        if force_reprocess:
            ecg_data = process_data(ecg_data)
            ecg_data.to_pickle(os.path.join(dataset_path, f"training/filtered_{f_name}.pk"))

        return ecg_data
    except (OSError, FileNotFoundError, Exception):
        return


def load_dataset(save_name="dataframe.pk", force_reload=False, process=True, force_reprocess=False, ecg_range=None, ecg_meas_diag=None):
    dataset = None
    if not force_reload:
        dataset = load_dataset_pickle(process, save_name, force_reprocess)
    if dataset is None:
        print("Failed to load from pickle, regenerating files")
        dataset = load_dataset_scratch(process, ecg_range, ecg_meas_diag, save_name=save_name)
    """
    else:
        dataset = filter_dataset(dataset[0], dataset[1], ecg_range, ecg_meas_diag)
    """

    return dataset


def update_dataset(f, save_name="dataframe.pk"):
    # Loads and updates a saved dataset with new columns etc, from function f
    dataset = load_dataset(save_name)
    dataset = f(dataset)
    dataset.to_pickle(os.path.join(dataset_path, f"training/filtered_{save_name}.pk"))
