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
dataset_name = "cinc_2017"  # One of cinc_2020, cinc_2027, safer_feas1

dataset_input_name = "dataframe_20_Jun"  # "dataframe_2" for cinC 2020, "dataframe" for safer feas1 and safer feas2 and "database" for cinc 2017
dataset_output_name = "20_Jun_cinc_2017_from_scratch"

test_size = 0.15
val_size = 0.15
# =======

if __name__ == '__main__':
    if dataset_name == "cinc_2020":
        df = CinC2020Dataset.load_dataset(save_name=dataset_input_name, force_reprocess=True)

        # At the moment we only select data with length between 3000 and 5000 which
        # can be truncated to 3000 samples (10s)
        def select_length(df):
            df_within_range = df[(df["length"] <= 5000) & (df["length"] >= 3000)].copy()
            df_within_range["data"] = df_within_range["data"].map(lambda x: x[:3000])
            df_within_range["length"] = df_within_range["data"].map(lambda x: x.shape[0])
            return df_within_range

        df = select_length(df)

        # CinC 2020 dataset
        val_dataset = df[df["dataset"] == "cpsc_2018"]
        train_dataset, test_dataset = train_test_split(df[df["dataset"] != "cpsc_2018"], test_size=test_size, stratify=df[df["dataset"] != "cpsc_2018"]["class_index"])

        print(train_dataset["class_index"].value_counts())
        print(test_dataset["class_index"].value_counts())
        print(val_dataset["class_index"].value_counts())

        # Save the CinC2017 data splits for consistent results!
        train_dataset.to_pickle(f"TrainedModels/{dataset_output_name}_train.pk")
        test_dataset.to_pickle(f"TrainedModels/{dataset_output_name}_test.pk")
        val_dataset.to_pickle(f"TrainedModels/{dataset_output_name}_val.pk")


    elif dataset_name == "cinc_2017":
        cinc2017_df = CinCDataset.load_cinc_dataset(dataset_input_name)
        cinc2017_df = cinc2017_df[cinc2017_df["length"] == 9000]
        cinc2017_df["measDiag"].value_counts()
        cinc2017_df = cinc2017_df[cinc2017_df["measDiag"] != DiagEnum.PoorQuality]

        def class_index_map(diag):
            if diag == DiagEnum.NoAF:
                return 0
            elif diag == DiagEnum.AF:
                return 1
            elif diag == DiagEnum.CannotExcludePathology:
                return 2
            elif diag == DiagEnum.Undecided:
                return 0

        cinc2017_df["class_index"] = cinc2017_df["measDiag"].map(class_index_map)

        train_dataset_2017, test_val = train_test_split(cinc2017_df.dropna(subset="class_index"),
                                                        test_size=test_size + val_size,
                                                        stratify=cinc2017_df["class_index"].dropna())

        test_dataset_2017, val_dataset_2017 = train_test_split(test_val,
                                                               test_size=val_size / (test_size + val_size),
                                                               stratify=test_val["class_index"])

        print(train_dataset_2017["class_index"].value_counts())
        print(test_dataset_2017["class_index"].value_counts())
        print(val_dataset_2017["class_index"].value_counts())

        train_dataset_2017.to_pickle(f"TrainedModels/{dataset_output_name}_train.pk")
        test_dataset_2017.to_pickle(f"TrainedModels/{dataset_output_name}_test.pk")
        val_dataset_2017.to_pickle(f"TrainedModels/{dataset_output_name}_val.pk")


    elif dataset_name == "safer_feas1":
        pt_data = SAFERDataset.load_pt_dataset(1)
        ecg_data = SAFERDataset.load_ecg_csv(1, pt_data, ecg_range=None,
                                             ecg_meas_diag=None,
                                             feas2_offset=10000,
                                             feas2_ecg_offset=200000)

        ecg_data["feas"] = 1
        ecg_data["length"] = 9120
        ecg_data["rri_len"] = 20

        def generate_patient_splits(pt_data, test_frac, val_frac):
            train_patients = []
            test_patients = []
            val_patients = []

            test_val_frac = test_frac + val_frac
            val_second_frac = val_frac / test_val_frac

            for val, df in pt_data.groupby("noAFRecs"):
                print(f"processing {val}")
                print(f"number of patients {len(df.index)}")

                n = math.floor(len(df.index) * test_val_frac)
                if test_val_frac > 0:
                    res = ((len(df.index) * test_val_frac) - n) / test_val_frac
                else:
                    res = 0
                n += np.random.binomial(res, test_val_frac)
                test_val = df.sample(n)

                n = math.floor(len(test_val.index) * val_second_frac)
                if val_second_frac > 0:
                    res = ((
                                       len(test_val.index) * val_second_frac) - n) / val_second_frac
                else:
                    res = 0
                n += np.random.binomial(res, val_second_frac)
                val = test_val.sample(n)
                val_patients.append(val)

                test_patients.append(test_val[~test_val["ptID"].isin(val["ptID"])])
                train_patients.append(df[~df["ptID"].isin(test_val["ptID"])])

            train_pt_df = pd.concat(train_patients)
            test_pt_df = pd.concat(test_patients)
            val_pt_df = pd.concat(val_patients)

            return train_pt_df, test_pt_df, val_pt_df


        pt_data, ecg_data = prepare_safer_data(pt_data, ecg_data)
        train_pts, test_pts, val_pts = generate_patient_splits(pt_data, test_size, val_size)

        print(f"Test AF: {test_pts['noAFRecs'].sum()} Normal: {test_pts['noNormalRecs'].sum()} Other: {test_pts['noOtherRecs'].sum()}")
        print(f"Train AF: {train_pts['noAFRecs'].sum()} Normal: {train_pts['noNormalRecs'].sum()} Other: {train_pts['noOtherRecs'].sum()}")
        print(f"Val AF: {val_pts['noAFRecs'].sum()} Normal: {val_pts['noNormalRecs'].sum()} Other: {val_pts['noOtherRecs'].sum()}")

        train_pts.to_pickle(os.path.join(constants.feas1_path, f"ECGs/{dataset_output_name}_train_pts.pk"))
        test_pts.to_pickle(os.path.join(constants.feas1_path, f"ECGs/{dataset_output_name}_test_pts.pk"))
        val_pts.to_pickle(os.path.join(constants.feas1_path, f"ECGs/{dataset_output_name}_val_pts.pk"))

        # We save the test and validation portions as ECGs because these are much smaller - note this takes a while though
        ecg_data_full, pt_data_full = load_feas1_chunk_range(input_name=dataset_input_name)
        pt_data_full, ecg_data_full = prepare_safer_data(pt_data_full, ecg_data_full)

        ecg_data_test = ecg_data_full[ecg_data_full["ptID"].isin(test_pts["ptID"])]
        ecg_data_val = ecg_data_full[ecg_data_full["ptID"].isin(val_pts["ptID"])]

        ecg_data_test.to_pickle(os.path.join(constants.feas1_path, f"ECGs/{dataset_output_name}_test.pk"))
        ecg_data_val.to_pickle(os.path.join(constants.feas1_path, f"ECGs/{dataset_output_name}_val.pk"))


    elif dataset_name == "safer_feas2":
        # We don't train on feas2 so just select all useful ECGs
        feas2_pt_data, feas2_ecg_data = SAFERDataset.load_feas_dataset(2, dataset_input_name)

        feas2_ecg_data["feas"] = 2
        feas2_ecg_data.index = feas2_ecg_data["measID"]

        feas2_pt_data, feas2_ecg_data = prepare_safer_data(feas2_pt_data, feas2_ecg_data)
        feas2_ecg_data.to_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_output_name}_af_processed.pk"))

