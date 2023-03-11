import scipy
from scipy import signal
import numpy as np

warp_filter = signal.butter(3, 0.005, 'lowpass', output='sos')
tukey_window = signal.windows.tukey(3000)


def temporal_warp(sig, r_peaks=None, warp_std=300):
    warp_field = np.random.randn(sig.shape[0]) * warp_std
    warp_field_smooth = signal.sosfiltfilt(warp_filter, warp_field, padlen=750)
    warp_field_smooth *= tukey_window

    warp_field_smooth += np.arange(sig.shape[0])
    warp_field_smooth = np.clip(warp_field_smooth, 0, sig.shape[0]-1)

    interp_f = scipy.interpolate.interp1d(np.arange(sig.shape[0]), sig, kind="cubic")

    new_r_peaks = None
    if r_peaks is not None:
        new_r_peaks = np.round(warp_field_smooth[r_peaks]).astype(int)

    return interp_f(warp_field_smooth), new_r_peaks