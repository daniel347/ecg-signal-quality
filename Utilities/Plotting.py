from scipy import signal
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def plot_ecg_spectrogram(x, fs=300, n_split=1, cut_range=None):
    sample_len = x.shape[0]
    time_axis = np.arange(sample_len)/fs

    freq_axis, time_axis, stft = signal.stft(x, nperseg=128, noverlap=3*128/4)
    time_axis = time_axis/fs
    freq_axis = freq_axis * fs
    stft = np.log(np.abs(stft))

    if cut_range is not None:
        stft = stft[cut_range[0]:cut_range[1], :]
        freq_axis = freq_axis[cut_range[0]:cut_range[1]]

    cuts = np.round(np.linspace(0, stft.shape[-1]-1, n_split+1)).astype(int)

    fig, ax = plt.subplots(n_split, 1, figsize=(12, 5), squeeze=False)
    for j in range(n_split):
        ax[j][0].imshow(np.flipud(stft[:, cuts[j]:cuts[j+1]]), extent=[time_axis[cuts[j]], time_axis[cuts[j+1]], freq_axis[0], freq_axis[-1]], aspect="auto")
        ax[-1][0].set_xlabel("Time")


def plot_ecg_poincare(rri):
    fig = plt.figure()
    plt.plot(rri[1:], rri[:-1])
    plt.xlabel("RR interval n")
    plt.ylabel("RR interval n-1")
    plt.show()


def plot_ecg_drr(rri):
    fig = plt.figure()
    drri = np.diff(rri)
    plt.plot(rri[1:], drri)
    plt.xlabel("RR interval")
    plt.ylabel("dRR interval")
    plt.show()


def plot_ecg(x, fs=300, n_split=1, r_peaks=None, attention=None, num_segments=None, figsize=(12, 5), export_quality=False):
    sample_len = x.shape[0]
    time_axis = np.arange(sample_len)/fs

    cuts = np.round(np.linspace(0, sample_len-1, n_split+1)).astype(int)

    y_step = 1

    fig, ax = plt.subplots(n_split, 1, figsize=figsize, squeeze=False, dpi=(250 if export_quality else 75))
    for j in range(n_split):
        ax[j][0].plot(time_axis[cuts[j]:cuts[j+1]], x[cuts[j]:cuts[j+1]])

        if r_peaks is not None:
            ax[j][0].plot(time_axis[r_peaks], x[r_peaks], "x")

        if attention is not None:
            print("Plotting attention")
            attention_step = (sample_len-1)/num_segments
            attention_gain = 0.5/attention.max()
            alpha = attention_gain * attention

            for i in range(num_segments):
                ax[j][0].axvspan(time_axis[math.floor(i*attention_step)],
                                 time_axis[math.floor((i+1)*attention_step)], color='green',
                                 alpha=alpha[i])

        ax[-1][0].set_xlabel("Time (s)")
        ax[j][0].set_xlim((time_axis[cuts[j]], time_axis[cuts[j+1]]))

        t_s = time_axis[cuts[j]]
        t_f = time_axis[cuts[j+1]]
        print(t_s, t_f)

        time_ticks = np.arange(t_s - t_s%0.2, t_f + (0.2 - t_f%0.2), 0.2)
        # Clip the ticks to just the range, to avoid whitespace at the edges of the signal (or duplicate times)
        time_ticks = time_ticks[np.logical_and(time_ticks > t_s, time_ticks < t_f)]

        decimal_labels = ~np.isclose(time_ticks, np.round(time_ticks))
        time_labels = np.round(time_ticks).astype(int).astype(str)
        time_labels[decimal_labels] = ""

        ax[j][0].set_ylim((x.min()-y_step, x.max()+y_step))

        ax[j][0].set_xticks(time_ticks, time_labels)
        ax[j][0].set_yticklabels([])


        ax[j][0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[j][0].yaxis.set_minor_locator(AutoMinorLocator(5))

        ax[j][0].grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax[j][0].grid(which='minor', linestyle='-', linewidth='0.5', color='lightgray')

    fig.tight_layout()