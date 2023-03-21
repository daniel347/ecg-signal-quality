import pandas as pd
import requests

import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import numpy as np
import scipy
from multiprocesspandas import applyparallel

import wfdb

base_url = "https://physionet.org/files/icentia11k-continuous-ecg/1.0/"
base_path = "../Datasets/Icentia11k/"


def download_file(args):
    url, path = args
    try:
        r = requests.get(url, params={"download":True})
        if r.status_code != 200:
            print(f"Error: {r.reason}")
            return
        with open(path, "wb") as f:
            f.write(r.content)
    except requests.exceptions.RequestException as e:
        print(e)


def find_files(patient_range, types=["atr", "hea", "dat"], n_per_patient = 50):
    urls = []
    filepaths = []
    record_names = []

    for i in range(patient_range[0], patient_range[1]):
        p_num = str(i).rjust(5, "0")
        path_ext = f"p{p_num[:2]}/p{p_num}/"

        # make the patient directories
        if not os.path.isdir(os.path.join(base_path, f"p{p_num[:2]}")):
            os.mkdir(os.path.join(base_path, f"p{p_num[:2]}"))
        if not os.path.isdir(os.path.join(base_path, f"p{p_num[:2]}", f"p{p_num}")):
            os.mkdir(os.path.join(base_path, f"p{p_num[:2]}", f"p{p_num}"))

        for j in range(n_per_patient):
            s_num = str(j).rjust(2, "0")
            record_names.append(os.path.join(base_path,
                                               f"p{p_num[:2]}",
                                               f"p{p_num}",
                                               f"p{p_num}_s{s_num}"))
            for t in types:
                section_file = f"p{p_num}_s{s_num}.{t}"
                record_url = base_url + path_ext + section_file
                record_filepath = os.path.join(base_path,
                                               f"p{p_num[:2]}",
                                               f"p{p_num}",
                                               f"p{p_num}_s{s_num}.{t}")

                urls.append(record_url)
                filepaths.append(record_filepath)

    return urls, filepaths, record_names


def download_parallel(urls, filepaths):
    cpus = cpu_count()
    for _ in tqdm(ThreadPool(cpus - 1).imap_unordered(download_file, zip(urls, filepaths)), total=len(urls)):
        pass


def split_recording(data, cut_len):
    cut_samp = cut_len * data.fs
    splits = np.arange(0, data.data.shape[0], cut_samp)

    data_splits = []

    afib = False
    aflutter = False

    # Note here the split array includes 0 so we get an empty array in data_splits[0],
    # which is filtered so the rest of the diags align with the data

    for i, (start, end) in enumerate(zip(splits, splits[1:])):
        ann_notes = data.notes[np.logical_and(data.r_peaks >= start, data.r_peaks < end)]

        if "(N" in ann_notes:
            afib = False
            aflutter = False
        elif "(AFIB" in ann_notes:
            afib = True
            aflutter = False
        elif "(AFL" in ann_notes:
            afib = False
            aflutter = True

        data_split = data.copy()
        data_split.data = data_split.data[start:end]
        data_split.r_peaks = data_split.r_peaks[np.logical_and(data.r_peaks >= start, data.r_peaks < end)] - start
        data_split.notes = ann_notes
        data_split.annotations = data.symbols[np.logical_and(data.r_peaks >= start, data.r_peaks < end)]

        if afib:
            data_split["class_index"] = 1
        elif aflutter:
            data_split["class_index"] = 2
        else:
            data_split["class_index"] = 0

        data_split["length"] = cut_samp
        data_splits.append(data_split)

    return pd.DataFrame(data_splits)


def load_dataset_scratch(record_names):
    long_df = []

    for n in record_names:
        try:
            ann = wfdb.rdann(n, "atr")
            rec = wfdb.rdrecord(n)

            data = rec.p_signal[:, 0]
            adc_gain = rec.adc_gain[0]
            fs = rec.fs

            r_peaks = np.array(ann.sample)
            symbols = np.array(ann.symbol)
            notes = np.array(ann.aux_note)

            ptID = int(rec.file_name[0][1:6])
            sample_id = int(rec.file_name[0][8:10])

            long_df.append(pd.Series({"data": data,
                                      "adc_gain": adc_gain,
                                      "fs": fs,
                                      "r_peaks": r_peaks,
                                      "symbols": symbols,
                                      "notes": notes,
                                      "ptID": ptID,
                                      "sampleID": sample_id}))
        except (FileNotFoundError, FileExistsError):
            print(f"Error reading {n}")

    print("Completed reading")

    long_df = pd.DataFrame(long_df)
    print(long_df.head())

    print("Processing Data")
    long_df = process_data(long_df)
    print(long_df.head())

    short_dfs = []

    print("Cutting to short segments")
    for i, long_sample in long_df.iterrows():
        short_dfs.append(split_recording(long_sample, 10))

    short_df = pd.concat(short_dfs, ignore_index=True)
    print(short_df.head())
    return short_dfs


def process_data(ecg_data, f_low=0.67, f_high=30, resample_rate=300):
    # perform band pass filtering, notch filtering (for power line interference)

    fs_list = pd.unique(ecg_data["fs"])
    bandpass_dict = {fs: scipy.signal.butter(3, [f_low, f_high], 'bandpass', fs=fs, output='sos') for fs in fs_list}
    notch_dict = {fs: scipy.signal.butter(3, [48, 52], 'bandstop', fs=fs, output='sos') for fs in fs_list}

    def filter_and_norm(x, sos):
        import scipy
        x_filt = scipy.signal.sosfiltfilt(sos, x, padlen=150)
        x_norm = (x_filt - x_filt.mean()) / x_filt.std()
        return x_norm

    ecg_data["data"] = ecg_data.apply(lambda x: filter_and_norm(x["data"], bandpass_dict[x["fs"]]), axis=1)
    ecg_data["data"] = ecg_data.apply(lambda x: filter_and_norm(x["data"], notch_dict[x["fs"]]), axis=1)

    def resample(x, orig_fs):
        import scipy
        resample_len = int(round(x.shape[-1] * resample_rate/orig_fs))
        return x if (x.shape[-1] == resample_len) else scipy.signal.resample(x, resample_len)

    ecg_data["data"] = ecg_data.apply(lambda x: resample(x["data"], x["fs"]), axis=1)
    ecg_data["length"] = ecg_data["data"].map(lambda x: x.shape[-1])
    ecg_data["fs"] = resample_rate

    # Scale the R peak positions to the new samplerate
    ecg_data["r_peaks"] = ecg_data["r_peaks"] * resample_rate/ecg_data["fs"]

    return ecg_data


if __name__ == "__main__":
    urls, filepaths, record_names = find_files([0, 100], types=["atr", "dat", "hea"], n_per_patient=1)
    # download_parallel(urls, filepaths)

    ecg_df = load_dataset_scratch(record_names)
    print("hi")

