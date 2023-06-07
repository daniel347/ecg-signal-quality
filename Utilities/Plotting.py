from scipy import signal
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from sklearn.metrics import ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_ecg_spectrogram(x, fs=300, n_split=1, cut_range=None, figsize=(12, 5), export_quality=False):
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

    fig, ax = plt.subplots(n_split, 1, figsize=figsize, squeeze=False, dpi=(250 if export_quality else 75))
    for j in range(n_split):
        ax[j][0].imshow(np.flipud(stft[:, cuts[j]:cuts[j+1]]), extent=[time_axis[cuts[j]], time_axis[cuts[j+1]], freq_axis[0], freq_axis[-1]], aspect="auto")
        ax[-1][0].set_xlabel("Time (s)")
        ax[j][0].set_ylabel("Frequency (Hz)")

    fig.tight_layout()


def plot_ecg_poincare(rri, rri_len, figsize=(5, 5)):
    fig = plt.figure(figsize=figsize)
    rri_non_zero = rri[:rri_len]
    plt.plot(rri_non_zero[:-1], rri_non_zero[1:], "o")
    plt.xlabel("RR interval n")
    plt.ylabel("RR interval n+1")
    plt.xlim((0.3, 1.5))
    plt.ylim((0.3, 1.5))
    plt.show()


def plot_ecg_drr(rri, rri_len, figsize=(4, 4), export_quality=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=(250 if export_quality else 75))
    drri = np.diff(rri[:rri_len])
    ax.plot(rri[1:rri_len], drri, "o")
    ax.set_xlabel("RR")
    ax.set_ylabel("dRR")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_mat, labels):
    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay(conf_mat, display_labels=labels).plot(cmap="GnBu", ax=ax, colorbar=False)
    fig.tight_layout()
    plt.show()

def plot_confusion_matrix_2(confusion_matrix, labels, colour='YlOrBr'):
    # Normalize the confusion matrix
    normalized_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(normalized_matrix, cmap=colour, alpha=0.8)

    # Set labels, title, and colorbar
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.colorbar(im)

    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="w" if normalized_matrix[i, j] > 0.5 else "k")

    # Show the plot
    fig.tight_layout()
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
            r_peaks = r_peaks[np.logical_and(r_peaks < sample_len, r_peaks >= 0)]
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

        ax[-1][0].set_xlabel("Time (s)")  # Only set the last axis label
        ax[j][0].set_xlim((time_axis[cuts[j]], time_axis[cuts[j+1]]))

        t_s = time_axis[cuts[j]]
        t_f = time_axis[cuts[j+1]]

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

def plot_ecg_with_attention(x, fs=300, n_split=1, r_peaks=None, attention=None, num_segments=None, figsize=(12, 8), export_quality=False, save_name=None):
    sample_len = x.shape[0]
    time_axis = np.arange(sample_len)/fs

    cuts = np.round(np.linspace(0, sample_len-1, n_split+1)).astype(int)
    n_heads = 0
    if attention is not None:
        n_heads = attention.shape[0]

    y_step = 1

    fig, ax = plt.subplots(n_split * (n_heads + 1), 1, figsize=figsize, squeeze=False, dpi=(250 if export_quality else 75))
    for j in range(n_split):
        ax[j*(n_heads + 1)][0].plot(time_axis[cuts[j]:cuts[j+1]], x[cuts[j]:cuts[j+1]])

        if r_peaks is not None:
            r_peaks = r_peaks[np.logical_and(r_peaks < sample_len, r_peaks >= 0)]
            ax[j*(n_heads + 1)][0].plot(time_axis[r_peaks], x[r_peaks], "x")

        if attention is not None:
            print("Plotting attention")

            att_start = int(round((attention.shape[-1]/n_split) * j))
            att_end = int(round((attention.shape[-1]/n_split) * (j + 1)))

            for h in range(n_heads):
                ax[j*(n_heads + 1) + 1 + h][0].imshow(attention[h, :, att_start:att_end], aspect="auto")
                ax[j*(n_heads + 1) + 1 + h][0].set_axis_off()

        ax[-(n_heads + 1)][0].set_xlabel("Time (s)")  # Only set the last axis label
        ax[j*(n_heads + 1)][0].set_xlim((time_axis[cuts[j]], time_axis[cuts[j+1]]))

        t_s = time_axis[cuts[j]]
        t_f = time_axis[cuts[j+1]]

        time_ticks = np.arange(t_s - t_s%0.2, t_f + (0.2 - t_f%0.2), 0.2)
        # Clip the ticks to just the range, to avoid whitespace at the edges of the signal (or duplicate times)
        time_ticks = time_ticks[np.logical_and(time_ticks > t_s, time_ticks < t_f)]

        decimal_labels = ~np.isclose(time_ticks, np.round(time_ticks))
        time_labels = np.round(time_ticks).astype(int).astype(str)
        time_labels[decimal_labels] = ""

        ax[j*(n_heads + 1)][0].set_ylim((x.min()-y_step, x.max()+y_step))

        ax[j*(n_heads + 1)][0].set_xticks(time_ticks, time_labels)
        ax[j*(n_heads + 1)][0].set_yticklabels([])


        ax[j*(n_heads + 1)][0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[j*(n_heads + 1)][0].yaxis.set_minor_locator(AutoMinorLocator(5))

        ax[j*(n_heads + 1)][0].grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax[j*(n_heads + 1)][0].grid(which='minor', linestyle='-', linewidth='0.5', color='lightgray')

    fig.tight_layout()