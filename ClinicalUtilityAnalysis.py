from DataHandlers.SAFERDatasetV2 import SaferDataset
import pandas as pd
from DataHandlers.DiagEnum import DiagEnum

def filter_ecgs(pt, ecg):
    ecg_new = ecg[ecg.length == 9120]
    ecg_new = ecg_new[ecg_new.measDiag != DiagEnum.Undecided]
    pt_new = pt[pt.ptID.isin(ecg_new.ptID)]

    return pt_new, ecg_new


def label_noise(x):
    return int(x == DiagEnum.PoorQuality)


results = pd.read_pickle("noise_predictions.pk")
dataset = SaferDataset(feas=1, label_gen=label_noise, filter_func=filter_ecgs)

ecg_df = dataset.ecg_data
ecg_df["prediction"] = results

ecg_df = ecg_df[ecg_df.measDiagAgree | (ecg_df.measDiagRev1 == "DiagEnum.Undecided") | (ecg_df.measDiagRev2 == "DiagEnum.Undecided")]

ecgs_reviewed = ((ecg_df.measDiag != DiagEnum.Undecided) & (ecg_df.prediction < 0)).sum()
total_ecgs_reviewed = (ecg_df.measDiag != DiagEnum.Undecided).sum()

print(f"ECGs reviewed: {ecgs_reviewed}")
print(f"Total ECGs reviewed: {total_ecgs_reviewed}")

ecg_df["isAF"] = ecg_df.measDiag == DiagEnum.AF

pts_found_af = ecg_df[ecg_df.prediction < 0].groupby("ptID").isAF.sum()
total_pts_found_af = ecg_df.groupby("ptID").isAF.sum()

print(f"AF pts found: {(pts_found_af > 0).sum()}")
print(f"Total AF pts reviewed: {(total_pts_found_af > 0).sum()}")

ecg_df["commentMentionsPoorQuality"] = ecg_df["measComm"].str.contains("poor quality", case=False)
ecg_df["commentMentionsPoorQuality"] = ecg_df["commentMentionsPoorQuality"].fillna(False)

poor_qual_pred_counts = (ecg_df[ecg_df.commentMentionsPoorQuality].prediction > 0).value_counts()
print(f"prediction counts for labelled poor quality: {poor_qual_pred_counts}")


