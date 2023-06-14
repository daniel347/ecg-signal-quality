import pandas as pd
import numpy as np

mit_dataset_path = "Datasets/mit-bih-noise-stress-test-database"


def split_signal(data, split_len):
    data_splits = []
    splits = np.arange(0, data["data"].shape[0], split_len)

    for i, (start, end) in enumerate(zip(splits, splits[1:])):
        data_split = data.copy()
        data_split["data"] = data["data"][start:end]
        data_split["data"] = (data_split["data"] - data_split["data"].mean())/ data_split["data"].std()

        data_split.name = i
        data_splits.append(data_split)

    return data_splits


def load_noise_segments(noises, split_length, f_low=0.67, f_high=30, fs=300):
    noise_dfs = []

    for n_path in noises:
        rec = wfdb.rdrecord(os.path.join(mit_dataset_path, n_path))
        sig = np.concatenate([rec.p_signal[:, 0], rec.p_signal[:, 1]])

        bandpass = signal.butter(3, [f_low, f_high], 'bandpass', fs=rec.fs,
                                 output='sos')
        notch = signal.butter(3, [48, 52], 'bandstop', fs=rec.fs, output='sos')

        sig = filter_and_norm(sig, bandpass)
        sig = filter_and_norm(sig, notch)

        sig = resample(sig, rec.fs, fs)
        sig_series = pd.Series(
            data={"data": sig, "fs": fs, "noise_type": n_path})

        split_signals = split_signal(sig_series, split_length)
        split_signals = pd.DataFrame(split_signals)

        noise_dfs.append(split_signals)

    noise_df = pd.concat(noise_dfs, ignore_index=True)
    return noise_df