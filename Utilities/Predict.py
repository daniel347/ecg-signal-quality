import torch
import pandas as pd
import numpy as np

def get_predictions_transformer(model, dataloader, dataset, device):
    model.eval()

    true_labels = []
    predictions = []

    outputs = []
    inds = []

    with torch.no_grad():
        for i, (signals, labels, ind) in enumerate(dataloader):
            signal = signals[0].to(device).float()
            rris = signals[1].to(device).float()
            rri_len = signals[2].to(device).float()

            labels = labels.long().detach().numpy()
            true_labels.append(labels)

            output = model(signal, rris, rri_len).detach().to("cpu").numpy()

            prediction = output
            predictions.append(prediction)

            for i, o in zip(ind, output):
                outputs.append(o)
                if isinstance(i, str):
                    inds.append(i)
                else:
                    inds.append(i.item())

    dataset["prediction"] = pd.Series(data=outputs, index=inds)

    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)

    return predictions, true_labels

def F1_ind(conf_mat, ind):
    return (2 * conf_mat[ind, ind])/(np.sum(conf_mat[ind]) + np.sum(conf_mat[:, ind]))

def print_results(conf_mat):
    print("Confusion matrix:")
    print(conf_mat)

    print(f"Sensitivity: {conf_mat[1, 1]/np.sum(conf_mat[1]):0.3f}")
    print(f"Specificity: {(conf_mat[0, 0] + conf_mat[0, 2] + conf_mat[2, 0] + conf_mat[2, 2])/(np.sum(conf_mat[0]) + np.sum(conf_mat[2])):0.3f}")
    print("")

    print(f"Normal F1: {F1_ind(conf_mat, 0):0.3f}")
    print(f"AF F1: {F1_ind(conf_mat, 1):0.3f}")
    print(f"Other F1: {F1_ind(conf_mat, 2):0.3f}")