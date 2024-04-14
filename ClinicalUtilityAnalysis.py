from DataHandlers.SAFERDatasetV2 import SaferDataset
import pandas as pd
from DataHandlers.DiagEnum import DiagEnum
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Utilities.Predict import *
from Utilities.Plotting import *

def filter_ecgs(pt, ecg):
    ecg_new = ecg[ecg.length == 9120]
    ecg_new = ecg_new[ecg_new.measDiag != DiagEnum.Undecided]
    ecg_new = ecg_new[ecg_new.measDiagAgree |
                      (ecg_new.measDiagRev1 == DiagEnum.Undecided) |
                      (ecg_new.measDiagRev2 == DiagEnum.Undecided)]
    pt_new = pt[pt.ptID.isin(ecg_new.ptID)]

    return pt_new, ecg_new


def label_noise(x):
    return int(x == DiagEnum.PoorQuality)


def compute_metrics(ecg_df, pred_thresh, prediction_label):
    ecgs_reviewed = ((ecg_df.measDiag != DiagEnum.Undecided) & (ecg_df[prediction_label] < pred_thresh)).sum()
    total_ecgs_reviewed = (ecg_df.measDiag != DiagEnum.Undecided).sum()

    ecg_df["isAF"] = ecg_df.measDiag == DiagEnum.AF

    pts_found_af = ecg_df[ecg_df[prediction_label] < pred_thresh].groupby("ptID").isAF.sum()
    total_pts_found_af = ecg_df.groupby("ptID").isAF.sum()

    ecg_df["commentMentionsPoorQuality"] = ecg_df["measComm"].str.contains("poor quality", case=False)
    ecg_df["commentMentionsPoorQuality"] = ecg_df["commentMentionsPoorQuality"].fillna(False)

    poor_qual_pred_counts = (ecg_df[ecg_df.commentMentionsPoorQuality][prediction_label] > pred_thresh).value_counts()

    return ecgs_reviewed, total_ecgs_reviewed, pts_found_af, total_pts_found_af, poor_qual_pred_counts

print("About to load predictions")
feas = 3
results = pd.read_pickle(f"noise_predictions_{feas}.pk")
dataset = SaferDataset(feas=feas, label_gen=label_noise, filter_func=filter_ecgs)

ecg_df = dataset.ecg_data
ecg_df["prediction"] = results

"""
plt.hist(ecg_df[ecg_df.measDiag == DiagEnum.PoorQuality].prediction, label="poor quality", alpha=0.5)
plt.hist(ecg_df[ecg_df.measDiag != DiagEnum.PoorQuality].prediction, label="sufficient quality", alpha=0.5)
plt.legend()
plt.show()
"""

if feas != 1:
    conf_mat = confusion_matrix(ecg_df["class_index"],
                                ecg_df["prediction"] > 0)
    print("============================")
    print("CNN confusion matrix results")
    print_noise_results(conf_mat)
    print("============================")

conf_mat = confusion_matrix(ecg_df["class_index"],
                            ecg_df["tag_orig_Poor_Quality"])
print("============================")
print("Cardiolund confusion matrix results")
print_noise_results(conf_mat)
print("============================")

ecgs_reviewed, total_ecgs_reviewed, pts_found_af, total_pts_found_af, poor_qual_pred_counts = compute_metrics(ecg_df, 0, "prediction")

pts_found_greater_0 = pts_found_af[pts_found_af > 0]
patients_not_found_n_af = total_pts_found_af[~total_pts_found_af.index.isin(pts_found_greater_0.index)].value_counts()
patients_not_found_n_af = patients_not_found_n_af.drop([0])

plt.bar(patients_not_found_n_af.index, patients_not_found_n_af.values)
plt.show()


# Plot the misclassified ECGS
ecg_df["dataset_idx"] = np.arange(0, len(ecg_df.index))
false_positives = ecg_df[(ecg_df["prediction"] > 0) & (ecg_df["measDiag"] == DiagEnum.AF)]
for i, sig in false_positives.iterrows():
    ecg, _, _ = dataset[sig.dataset_idx]
    plot_ecg(ecg, n_split=3)
    print(sig[["measDiag", "ptID", "ptDiag", "prediction"]])
    plt.show()


print("===================================")
print("Results for thresh 0")
print(f"ECGs reviewed: {ecgs_reviewed}")
print(f"Total ECGs  reviewed: {total_ecgs_reviewed}")
print(f"AF pts found: {(pts_found_af > 0).sum()}")
print(f"Total AF pts reviewed: {(total_pts_found_af > 0).sum()}")
print(f"prediction counts for labelled poor quality: {poor_qual_pred_counts}")
print("===================================")

# Cardiolund results
ecgs_reviewed_cardiolund, _, pts_found_af_cardiolund, _, _ = compute_metrics(ecg_df, 0.5, "tag_orig_Poor_Quality")

print("===================================")
print("Results for Cardiolund")
print(f"ECGs reviewed: {ecgs_reviewed_cardiolund}")
print(f"Total ECGs  reviewed: {total_ecgs_reviewed}")
print(f"AF pts found: {(pts_found_af_cardiolund > 0).sum()}")
print(f"Total AF pts reviewed: {(total_pts_found_af > 0).sum()}")
print("===================================")


pred_threshes = np.linspace(-1.25, 1.25, 25)
p_thresh = 1/(1 + np.exp(-pred_threshes))

ecgs_reviewed = []
pts_found_af = []

for t in pred_threshes:
    ecgs_rev, _, pts_found, _, _ = compute_metrics(ecg_df, t, "prediction")
    ecgs_reviewed.append(ecgs_rev)
    pts_found_af.append((pts_found > 0).sum())

ecgs_reviewed = np.array(ecgs_reviewed)
pts_found_af = np.array(pts_found_af)

review_fraction_per_af = ecgs_reviewed/pts_found_af

fig, ax = plt.subplots()
ax.plot(pts_found_af, review_fraction_per_af, label="Ours")
ax.plot([(total_pts_found_af > 0).sum()], [total_ecgs_reviewed/(total_pts_found_af > 0).sum()], "go", label="No filter")
ax.plot([(pts_found_af_cardiolund > 0).sum()], [ecgs_reviewed_cardiolund/(pts_found_af_cardiolund > 0).sum()], "ro", label="Cardiolund")

ax.legend()
ax.set_ylim((0, 160))

print(total_ecgs_reviewed/(total_pts_found_af > 0).sum())

for x, y, p in zip(pts_found_af, review_fraction_per_af, p_thresh):
    ax.annotate(f"{p:.3f}", (x, y))

ax.set_ylabel("ECGs reviewed per AF patient")
ax.set_xlabel("Number of AF patients found")

plt.show()


