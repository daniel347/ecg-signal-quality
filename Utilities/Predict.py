import numpy as np


def F1_ind(conf_mat, ind):
    return (2 * conf_mat[ind, ind])/(np.sum(conf_mat[ind]) + np.sum(conf_mat[:, ind]))

def print_af_results(conf_mat):
    print("Confusion matrix:")
    print(conf_mat)

    print(f"Sensitivity: {conf_mat[1, 1]/np.sum(conf_mat[1]):0.3f}")
    print(f"Specificity: {(conf_mat[0, 0] + conf_mat[0, 2] + conf_mat[2, 0] + conf_mat[2, 2])/(np.sum(conf_mat[0]) + np.sum(conf_mat[2])):0.3f}")
    print("")

    print(f"Normal F1: {F1_ind(conf_mat, 0):0.3f}")
    print(f"AF F1: {F1_ind(conf_mat, 1):0.3f}")
    print(f"Other F1: {F1_ind(conf_mat, 2):0.3f}")


def print_noise_results(conf_mat):
    print("Confusion matrix:")
    print(conf_mat)

    print(f"Sensitivity: {conf_mat[1, 1] / np.sum(conf_mat[1]):0.3f}")
    print(f"Specificity: {conf_mat[0, 0] / np.sum(conf_mat[0]):0.3f}")
    print("")

    print(f"Sufficient quality F1: {F1_ind(conf_mat, 0):0.3f}")
    print(f"Insufficient quality F1: {F1_ind(conf_mat, 1):0.3f}")