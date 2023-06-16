import sys

from DataHandlers.DiagEnum import DiagEnum
import DataHandlers.SAFERDataset as SAFERDataset
import DataHandlers.CinC2020Dataset as CinC2020Dataset
import DataHandlers.CinC2020Enums
import DataHandlers.CinCDataset as CinCDataset
import DataHandlers.MITNSTDataset as MITNSTDataset
from DataHandlers.Dataloaders import prepare_safer_data
import os

import Utilities.constants as constants

from torch.utils.data import Dataset, DataLoader

from Models.SpectrogramTransformerAttentionPooling import TransformerModel
from torch.optim.lr_scheduler import LambdaLR

from Utilities.Training import *

import torch
from sklearn.model_selection import train_test_split
from Utilities.Predict import *

# A fudge because I moved the files
sys.modules["SAFERDataset"] = SAFERDataset
sys.modules["CinC2020Dataset"] = CinC2020Dataset
sys.modules["DiagEnum"] = DataHandlers.DiagEnum
sys.modules["CinC2020Enums"] = DataHandlers.CinC2020Enums
sys.modules["CinCDataset"] = CinCDataset


# ====  Options  ====
dataset_name = "safer_feas1"  # One of cinc_2020, cinc_2027, safer_feas1

dataset_input_name = "dataframe_2"
dataset_output_name = "16_Jun_safer_split_script_test"

test_size = 0.15
val_size = 0.15
# =======


if dataset_name == "cinc_2020":
    df = CinC2020Dataset.load_dataset(save_name=dataset_input_name)

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
    cinc2017_df = CinCDataset.load_cinc_dataset()
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

    pt_data, ecg_data = prepare_safer_data(pt_data, ecg_data)

    # TODO: Should this use the new system -
    #  Even though I did not use it in my experiments?
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

    train_pts.to_pickle(os.path.join(constants.feas1_path, f"{dataset_output_name}.pk"))
    test_pts.to_pickle(os.path.join(constants.feas1_path, f"{dataset_output_name}.pk"))
    val_pts.to_pickle(os.path.join(constants.feas1_path, f"{dataset_output_name}.pk"))
