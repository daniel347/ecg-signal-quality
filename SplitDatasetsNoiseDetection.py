import os
import pandas as pd
from sklearn.model_selection import train_test_split

from DataHandlers.DiagEnum import DiagEnum
import DataHandlers.SAFERDataset as SAFERDataset
import DataHandlers.CinC2020Dataset as CinC2020Dataset
import DataHandlers.CinCDataset as CinCDataset
from DataHandlers.Dataloaders import load_feas1_chunk_range, prepare_safer_data

import Utilities.constants as constants
from Utilities.Training import *
from Utilities.Predict import *


# ====  Options  ====
dataset_name = "safer_feas2"  # One of cinc_2020, cinc_2027, safer_feas1

dataset_input_name = "dataframe_20_Jun"  # "dataframe_2" for cinC 2020, "dataframe" for safer feas1 and safer feas2 and "database" for cinc 2017
dataset_output_name = "20_Jun_from_scratch"

test_size = 0.15
val_size = 0.15
# =======

if __name__ == '__main__':
    if dataset_name == "safer_feas2":
        feas2_pt_data, feas2_ecg_data = SAFERDataset.load_feas_dataset(2, dataset_input_name)

        feas2_ecg_data["feas"] = 2
        feas2_ecg_data.index = feas2_ecg_data["measID"]

        feas2_pt_data["noRecs"] = feas2_ecg_data["ptID"].value_counts()
        feas2_pt_data["noHQrecs"] = feas2_ecg_data[feas2_ecg_data["class_index"] == 0]["ptID"].value_counts()
        feas2_pt_data["noHQrecsNotUndecided"] = \
        feas2_ecg_data[(feas2_ecg_data["class_index"] == 0) & (feas2_ecg_data["measDiag"] != DiagEnum.Undecided)]["ptID"].value_counts()
        feas2_pt_data["noLQrecs"] = feas2_pt_data["noRecs"] - feas2_pt_data["noHQrecs"]
        feas2_pt_data[["noRecs", "noHQrecs", "noHQrecsNotUndecided"]] = feas2_pt_data[["noRecs", "noHQrecs", "noHQrecsNotUndecided"]].fillna(0)
        def make_SAFER_dataloaders(pt_data, ecg_data, test_frac, val_frac):

            train_patients = []
            test_patients = []
            val_patients = []

            lq_counts = np.array([0, 0, 0], dtype=int)
            total_counts = np.array([0, 0, 0], dtype=int)

            fracs = np.array([1 - test_frac - val_frac, test_frac, val_frac])

            total_lq_count = 0
            total_total_count = 0

            for val, pt in pt_data.iterrows():
                total_total_count += pt["noHQrecsNotUndecided"]
                total_lq_count += pt["noLQrecs"]

                exp_total_counts = total_total_count * fracs
                exp_lq_counts = total_lq_count * fracs

                loss_0 = np.sum(np.abs(lq_counts + np.array([pt["noLQrecs"], 0, 0]) - exp_lq_counts) + np.abs(
                    total_counts + np.array([pt["noHQrecsNotUndecided"], 0, 0]) - exp_total_counts))
                loss_1 = np.sum(np.abs(lq_counts + np.array([0, pt["noLQrecs"], 0]) - exp_lq_counts) + np.abs(
                    total_counts + np.array([0, pt["noHQrecsNotUndecided"], 0]) - exp_total_counts))
                loss_2 = np.sum(np.abs(lq_counts + np.array([0, 0, pt["noLQrecs"]]) - exp_lq_counts) + np.abs(
                    total_counts + np.array([0, 0, pt["noHQrecsNotUndecided"]]) - exp_total_counts))

                min_loss = min([loss_0, loss_1, loss_2])

                if min_loss == loss_0:
                    train_patients.append(pt)
                    lq_counts += np.array([pt["noLQrecs"], 0, 0], dtype=int)
                    total_counts += np.array([pt["noHQrecsNotUndecided"], 0, 0], dtype=int)
                elif min_loss == loss_1:
                    test_patients.append(pt)
                    lq_counts += np.array([0, pt["noLQrecs"], 0], dtype=int)
                    total_counts += np.array([0, pt["noHQrecsNotUndecided"], 0], dtype=int)
                else:
                    val_patients.append(pt)
                    lq_counts += np.array([0, 0, pt["noLQrecs"]], dtype=int)
                    total_counts += np.array([0, 0, pt["noHQrecsNotUndecided"]], dtype=int)

            train_pt_df = pd.DataFrame(train_patients)
            test_pt_df = pd.DataFrame(test_patients)
            val_pt_df = pd.DataFrame(val_patients)

            print(f"Test high quality: {test_pt_df['noHQrecsNotUndecided'].sum()} low quality: {test_pt_df['noLQrecs'].sum()} ")
            print(f"Train high quality: {train_pt_df['noHQrecsNotUndecided'].sum()} low quality: {train_pt_df['noLQrecs'].sum()} ")
            print(f"Val high quality: {val_pt_df['noHQrecsNotUndecided'].sum()} low quality: {val_pt_df['noLQrecs'].sum()}")


            train_dataset = None
            test_dataset = None
            val_dataset = None

            if not train_pt_df.empty:
                train_dataset = ecg_data[(ecg_data["ptID"].isin(train_pt_df["ptID"])) & (ecg_data["measDiag"] != DiagEnum.Undecided)]

            if not test_pt_df.empty:
                test_dataset = ecg_data[(ecg_data["ptID"].isin(test_pt_df["ptID"])) & (ecg_data["measDiag"] != DiagEnum.Undecided)]

            if not val_pt_df.empty:
                val_dataset = ecg_data[(ecg_data["ptID"].isin(val_pt_df["ptID"])) & (ecg_data["measDiag"] != DiagEnum.Undecided)]

            return train_dataset, test_dataset, val_dataset, test_pt_df, train_pt_df, val_pt_df


        train_dataset, test_dataset, val_dataset, test_pts, train_pts, val_pts = make_SAFER_dataloaders(feas2_pt_data,
                                                                                                              feas2_ecg_data,
                                                                                                              test_frac=test_size,
                                                                                                              val_frac=val_size)

        print(f"Test High quality: {test_pts['noHQrecsNotUndecided'].sum()} low quality: {test_pts['noLQrecs'].sum()}")
        print(f"Train High quality: {train_pts['noHQrecsNotUndecided'].sum()} low quality: {train_pts['noLQrecs'].sum()}")
        print(f"Val High quality: {val_pts['noHQrecsNotUndecided'].sum()} low quality: {val_pts['noLQrecs'].sum()}")

        train_dataset.to_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_output_name}_train.pk"))
        test_dataset.to_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_output_name}_test.pk"))
        val_dataset.to_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_output_name}_val.pk"))

        train_pts.to_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_output_name}_train_pts.pk"))
        test_pts.to_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_output_name}_test_pts.pk"))
        val_pts.to_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_output_name}_val_pts.pk"))

