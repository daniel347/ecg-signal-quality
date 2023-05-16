import numpy as np
from scipy.signal import windows, sosfiltfilt
import scipy.signal


def filter_and_norm(x, sos):
    from scipy.signal import sosfiltfilt
    x_filt = sosfiltfilt(sos, x, padlen=150)
    x_norm = (x_filt - x_filt.mean()) / x_filt.std()
    return x_norm


def resample(x, resample_rate, orig_fs):
    import scipy.signal
    resample_len = int(round(x.shape[-1] * resample_rate / orig_fs))
    return scipy.signal.resample(x, resample_len)

def get_rri_feature(x, n_beats=60):
    diffs = np.diff(x)[1:-1] # Discard first and last in case it is a false detect
    L = diffs.shape[0]
    if L > n_beats-1:
        return diffs[:n_beats-1]
    else:
        return np.pad(diffs, (0, n_beats-1-L), constant_values=0)


def get_r_peaks(x):
    import numpy as np
    import neurokit2 as nk

    try:
        _, info = nk.ecg_peaks(x["data"], sampling_rate=300)
    except (IndexError, Exception):
        # I think I get an index error if no R peaks are found
        return np.array([])
    return np.array(info["ECG_R_Peaks"])


def normalise_rri_feature(ecg_data):
    # Normalise the RRI features

    feature_norm = ecg_data.apply(lambda x: x["rri_feature"][:x["rri_len"]], axis=1)
    mean_feature = feature_norm.map(lambda x: x.mean()).mean()
    std_feature = feature_norm.map(lambda x: x.std()).mean()

    print(f"rri normalisation mean: {mean_feature} standard deviation: {std_feature}")

    return (ecg_data["rri_feature"] - mean_feature) / std_feature


def validate_r_peaks(x):
    try:
        if type(x) == np.ndarray:
            return True
        else:
            print(x)
            return False
    except(Exception):
        print(x)
        return False


def validate_data(x, l):
    try:
        return x.shape[0] == l
    except(Exception):
        return False

# Test to cut just the centre portion of SAFER data


def cut_ecg_center(ecg, cut_size):
    ecg_len = ecg["data"].shape[0]
    return ecg["data"][int((ecg_len - cut_size) / 2):-int((ecg_len - cut_size) / 2)]


def cut_ecg_adjust_r_peaks(ecg, cut_size):
    ecg_len = ecg["data"].shape[0]
    shifted_peaks = ecg["r_peaks_hamilton"] - int((ecg_len - cut_size) / 2)
    return shifted_peaks[np.logical_and(shifted_peaks >= 0, shifted_peaks < cut_size)]


def adaptive_gain_norm(x, w):
    x_mean_sub = np.pad(x - x.mean(), int((w - 1) / 2), "reflect")
    window = np.ones(w)
    sigma_square = np.convolve(x_mean_sub ** 2, window, mode="valid") / w
    gain = 1 / np.sqrt(sigma_square)

    return x * gain