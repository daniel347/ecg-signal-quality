import pandas as pd
import os
import scipy
import wfdb
import math
from parallel_pandas import ParallelPandas
from pandas.api.types import is_numeric_dtype
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset

from DataHandlers.DiagEnum import DiagEnum, feas1DiagToEnum, trialDiagToEnum
from DataHandlers.DataProcessUtilities import *

ParallelPandas.initialize(n_cpu=4, split_factor=4)

# Chunks the
chunk_size = 20000
num_chunks = math.ceil(162515 / chunk_size)

pt_offsets = [0, 10000, 20000]
ecg_offsets = [0, 300000, 600000]

feas2_path = r"Datasets\Feas2_DSiromani"
feas1_path = r"Datasets\Feas1_DSiromani"
trial_path = r"Datasets\Trial_DSiromani"

dataset_paths = [feas1_path, feas2_path, trial_path]


class SaferDataset(Dataset):

    def __init__(self, feas, label_gen=None, use_processed=True, ecg_range=None, ecg_meas_diag=None, filter_func=None):
        self.feas = feas
        self.dataset_path = dataset_paths[feas-1]

        self.use_processed = True
        self.label_gen = label_gen

        self.pt_data = pd.read_csv(os.path.join(self.dataset_path, "pt_data_anon_processed_paths.csv"))
        self.pt_data = self.pt_data[self.pt_data.processed_df.map(lambda x: isinstance(x, str))]
        self.pt_data.index = self.pt_data.ptID

        self.ecg_data = pd.read_csv(os.path.join(self.dataset_path, "rec_data_anon_processed.csv"))

        diag_enum_cols = ["measDiag", "measDiagRev1", "measDiagRev2",
                          "ptDiag", "ptDiagRev1", "ptDiagRev2", "ptDiagRev3"]
        for c in diag_enum_cols:
            self.ecg_data[c] = self.ecg_data[c].map(lambda x: DiagEnum[x.split(".")[1]])

        """
        # Turn string back to enum
        if isinstance(self.ecg_data.measDiag.dtype, str):
            self.ecg_data.measDiag = self.ecg_data.measDiag.map(lambda x: DiagEnum[x.split(".")[1]])
        elif is_numeric_dtype(self.ecg_data.measDiag.dtype):
            self.ecg_data.measDiag = self.ecg_data.measDiag.map(lambda x: DiagEnum(x))
            """

        self.ecg_data = self.ecg_data[self.ecg_data.ptID.isin(self.pt_data.ptID)]
        self.ecg_data = self.ecg_data.dropna(how="all", axis="columns")

        if filter_func:
            print(f"Diags before filter: {self.ecg_data.measDiag.value_counts()}")
            self.pt_data, self.ecg_data = filter_func(self.pt_data, self.ecg_data)
            print(f"Diags after filter: {self.ecg_data.measDiag.value_counts()}")

    def __len__(self):
        return len(self.ecg_data.index)

    def __getitem__(self, idx):
        ecg_row = self.ecg_data.iloc[idx]
        pt = self.pt_data.loc[ecg_row.ptID]
        pt_ecgs = pd.read_pickle(os.path.join(self.dataset_path, pt.processed_df))

        data_row = pt_ecgs.loc[ecg_row.measNo-1]  # -1 because I forgot to index starting at 1
        if self.label_gen:
            y = self.label_gen(data_row.measDiag)
            return data_row["data"], y, ecg_row.name

        else:
            return data_row["data"], ecg_row.name

def filter_dataset(pt_data, ecg_data, ecg_range, ecg_meas_diag):
    if ecg_range is not None:
        # We read only some of the ecgs - particularly for feas1 this breaks data into manageable chunks
        ecg_data = ecg_data[(ecg_data["measID"] >= ecg_range[0]) & (ecg_data["measID"] < ecg_range[1])]
        pt_data = pt_data[pt_data["ptID"].isin(ecg_data["ptID"])]

    if ecg_meas_diag is not None:
        ecg_data = ecg_data[ecg_data["measDiag"].isin(ecg_meas_diag)]
        pt_data = pt_data[pt_data["ptID"].isin(ecg_data["ptID"])]

    return pt_data, ecg_data


def generate_af_class_labels(dataset):
    """See notes/ emails for explaination of logic"""
    dataset["class_index"] = -1
    # print(f"cardiologists disagree {len(dataset.loc[~dataset['measDiagAgree']].index)}")

    # If not tagged assume Normal
    # print(f"Not tagged {len(dataset.loc[(dataset['not_tagged_ign_wide_qrs'] == 1) & (dataset['measDiag'] == DiagEnum.Undecided) & (dataset['measDiagAgree'])].index)}")
    dataset.loc[(dataset["not_tagged_ign_wide_qrs"] == 1) & (dataset["measDiag"] == DiagEnum.Undecided), "class_index"] = 0

    # The first review rejection is Normal (more or less)
    # print(f"Not first review rejection {len(dataset.loc[(dataset['not_tagged_ign_wide_qrs'] == 0) & (dataset['feas'] == 1) & (dataset['ffReview_sent'] == 1) & (dataset['ffReview_remain'] == 0)  & (dataset['measDiagAgree'])].index)}")
    dataset.loc[(dataset["not_tagged_ign_wide_qrs"] == 0) & (dataset["feas"] == 1) & (dataset["ffReview_sent"] == 1) & (dataset["ffReview_remain"] == 0), "class_index"] = 0

    # dataset.loc[(dataset["not_tagged_ign_wide_qrs"] == 0) & (dataset["feas"] == 2), "class_index"] = 2 # Anything tagged in feas 2 goes to other

    # print(f"AF by cardiologist {len(dataset.loc[(dataset['measDiag'] == DiagEnum.AF) & (dataset['measDiagAgree'])].index)}")
    dataset.loc[dataset["measDiag"] == DiagEnum.AF, "class_index"] = 1  # The cardiologist has said AF

    # Anything that's got this far is probably dodgy in some way
    # print(f"No AF/Heartblock by cardiologist {len(dataset.loc[(dataset['measDiag'].isin([DiagEnum.NoAF, DiagEnum.HeartBlock])) & (dataset['measDiagAgree'])].index)}")
    dataset.loc[dataset["measDiag"].isin([DiagEnum.NoAF, DiagEnum.HeartBlock]), "class_index"] = 2

    # print(f"poor quality/cep by cardiologist {len(dataset.loc[(dataset['measDiag'].isin([DiagEnum.CannotExcludePathology, DiagEnum.PoorQuality])) & (dataset['measDiagAgree'])].index)}")
    dataset.loc[(dataset["measDiag"].isin([DiagEnum.CannotExcludePathology, DiagEnum.PoorQuality])), "class_index"] = -1

    dataset["measDiagAgree"] = (dataset["measDiagRev1"] == dataset["measDiagRev2"]) |\
                               (dataset["measDiagRev1"] == DiagEnum.Undecided) |\
                               (dataset["measDiagRev2"] == DiagEnum.Undecided)
    dataset.loc[~dataset["measDiagAgree"], "class_index"] = -1

    # dataset = dataset[dataset["class_index"] != -1]

    return dataset[dataset["class_index"] != -1]


def add_ecg_class_counts(safer_pt_data, safer_ecg_data):
    safer_pt_data.index = safer_pt_data["ptID"]
    safer_pt_data["noNormalRecs"] = safer_ecg_data[safer_ecg_data["class_index"] == 0]["ptID"].value_counts()
    safer_pt_data["noAFRecs"] = safer_ecg_data[safer_ecg_data["class_index"] == 1]["ptID"].value_counts()
    safer_pt_data["noOtherRecs"] = safer_ecg_data[safer_ecg_data["class_index"] == 2]["ptID"].value_counts()

    safer_pt_data["noAFRecs"] = safer_pt_data["noAFRecs"].fillna(0)
    safer_pt_data["noNormalRecs"] = safer_pt_data["noNormalRecs"].fillna(0)
    safer_pt_data["noOtherRecs"] = safer_pt_data["noOtherRecs"].fillna(0)

    return safer_pt_data


def load_feas_dataset_scratch(process, feas, ecg_range, ecg_meas_diag, save_name, filter_func):
    dataset_path = dataset_paths[feas-1]

    pt_data = load_pt_dataset(feas)
    ecg_data = load_ecg_csv(feas)
    pt_data, ecg_data = filter_dataset(pt_data, ecg_data, ecg_range, ecg_meas_diag)

    if filter_func:
        pt_data, ecg_data = filter_func(pt_data, ecg_data)

    def read_pt_records(pt_data):
        ecg_dfs = []
        for _, x in tqdm(pt_data.iterrows(), total=len(pt_data.index)):
            ecg_dfs.append(read_pt_record(x, dataset_path))
        ecg_dfs = pd.concat(ecg_dfs, ignore_index=True)
        return ecg_dfs


    print("reading ECG files")
    ecg_data_and_adc_cols = read_pt_records(pt_data)
    ecg_data_and_adc_cols.index += ecg_offsets[feas-1]
    ecg_data_and_adc_cols = ecg_data_and_adc_cols.loc[ecg_data.index]

    ecg_data["data"] = ecg_data_and_adc_cols["data"]
    ecg_data["adc_gain"] = ecg_data_and_adc_cols["adc_gain"]

    ecg_data.dropna(subset=["data"], inplace=True)

    # Generate the class_index
    ecg_data["length"] = ecg_data["data"].map(lambda x: x.shape[-1])
    ecg_data.to_pickle(os.path.join(dataset_path, f"ptECGs/raw_{save_name}.pk"))

    if process:
        ecg_data = process_data(ecg_data)
        ecg_data.to_pickle(os.path.join(dataset_path, f"ptECGs/filtered_{save_name}.pk"))

    return pt_data, ecg_data


def process_data(feas2_ecg_data, f_low=0.67, f_high=30, resample_rate=300, parallel=True):
    # perform band pass filtering, notch filtering (for power line interference)
    sos = scipy.signal.butter(3, [f_low, f_high], 'bandpass', fs=500, output='sos')
    sos_notch = scipy.signal.butter(3, [48, 52], 'bandstop', fs=500, output='sos')

    def band_pass(x):
        from DataProcessUtilities import filter_and_norm
        return filter_and_norm(x, sos)
    def powerline_notch(x):
        from DataProcessUtilities import filter_and_norm
        return filter_and_norm(x, sos_notch)

    if parallel:
        feas2_ecg_data["data"] = feas2_ecg_data["data"].p_apply(band_pass)
        feas2_ecg_data["data"] = feas2_ecg_data["data"].p_apply(powerline_notch)
    else:
        feas2_ecg_data["data"] = feas2_ecg_data["data"].map(band_pass)
        feas2_ecg_data["data"] = feas2_ecg_data["data"].map(powerline_notch)

    if resample_rate != 500:
        def resampler(x):
            from DataProcessUtilities import resample
            return resample(x, resample_rate, 500)
        if parallel:
            feas2_ecg_data["data"] = feas2_ecg_data["data"].p_apply(resampler)
        else:
            feas2_ecg_data["data"] = feas2_ecg_data["data"].map(resampler)
        feas2_ecg_data["length"] = feas2_ecg_data["data"].map(lambda x: x.shape[-1])

    # Get beat positions and heartrate
    """
    def resampler(x):
        from DataProcessUtilities import get_r_peaks
        return get_r_peaks(x)
    """
    if parallel:
        feas2_ecg_data["r_peaks"] = feas2_ecg_data.p_apply(get_r_peaks, axis=1)
    else:
        feas2_ecg_data["r_peaks"] = feas2_ecg_data.apply(get_r_peaks, axis=1)
    """
    try:
        feas2_ecg_data["r_peaks"] = feas2_ecg_data.apply(get_r_peaks, axis=1)
    except:
        feas2_ecg_data["r_peaks"] = feas2_ecg_data.apply(get_r_peaks, axis=1)
    """

    feas2_ecg_data["heartrate"] = feas2_ecg_data.apply(lambda e: (len(e["r_peaks"]) / (e["length"] / resample_rate)) * 60, axis=1)

    # Get the rri feature
    feas2_ecg_data["rri_feature"] = (feas2_ecg_data["r_peaks"] / resample_rate).map(lambda x: get_rri_feature(x, 60))
    fewer_5_beats = feas2_ecg_data["rri_feature"].map(lambda x: np.sum(x == 0) > 55)
    # feas2_ecg_data = feas2_ecg_data[~fewer_5_beats]
    feas2_ecg_data.loc[:, "rri_len"] = feas2_ecg_data["rri_feature"].map(lambda x: x[x > 0].shape[-1])
    # feas2_ecg_data["rri_feature"] = normalise_rri_feature(feas2_ecg_data["rri_feature"])

    return feas2_ecg_data


def load_pt_dataset(feas):
    dataset_path = dataset_paths[feas-1]
    pt_df = pd.read_csv(os.path.join(dataset_path, "pt_data_anon.csv"))

    if feas in [1,2]:
        ecg_path_labels = "ptECGs/{:06d}/saferF{}_pt{:06d}"
        pt_df["file_path"] = pt_df["ptID"].map(lambda id: ecg_path_labels.format((id // 100) * 100, feas, id))
    elif feas == 3:
        ecg_path_labels = "ptECGs/{:06d}/saferT_pt{:06d}"
        pt_df["file_path"] = pt_df["ptID"].map(lambda id: ecg_path_labels.format((id // 100) * 100, id))

    pt_df["ptID"] += pt_offsets[feas-1]
    pt_df.index = pt_df["ptID"]
    pt_df["feas"] = feas


    return pt_df


def load_ecg_csv(feas):
    dataset_path = dataset_paths[feas-1]
    ecg_data = pd.read_csv(os.path.join(dataset_path, "rec_data_anon.csv"))
    ecg_data["ptID"] += pt_offsets[feas-1]
    ecg_data["data"] = None
    ecg_data["adc_gain"] = None

    ecg_data["measDiagAgree"] = (ecg_data["measDiagRev1"] == ecg_data["measDiagRev2"]) | \
                                (ecg_data["measDiagRev1"] == DiagEnum.Undecided) | \
                                (ecg_data["measDiagRev2"] == DiagEnum.Undecided)

    # convert data to the Enum
    diag_columns = ["ptDiag", "ptDiagRev1", "ptDiagRev2", "ptDiagRev3", "measDiag", "measDiagRev1", "measDiagRev2"]
    for diag_ind in diag_columns:
        if feas == 2:
            ecg_data[diag_ind] = ecg_data[diag_ind].map(lambda d: DiagEnum(d))
        elif feas == 1:
            ecg_data[diag_ind] = ecg_data[diag_ind].map(lambda d: feas1DiagToEnum(d))
        elif feas == 3:
            ecg_data[diag_ind] = ecg_data[diag_ind].map(lambda d: trialDiagToEnum(d))

    ecg_data["measIDpt"] = ecg_data["measID"]

    ecg_data["measID"] = ecg_data.index + ecg_offsets[feas-1]

    # Generate class index
    ecg_data["class_index"] = (ecg_data["measDiag"] == DiagEnum.PoorQuality).astype(int)
    ecg_data["length"] = None
    ecg_data.index = ecg_data["measID"]

    return ecg_data


def load_feas_dataset_pickle(process, f_name, feas=2, force_reprocess=False):
    dataset_path = dataset_paths[feas-1]

    try:
        pt_data = load_pt_dataset(feas)
        end_path = f"ptECGs/filtered_{f_name}.pk" if (process and not force_reprocess) else f"ptECGs/raw_{f_name}.pk"
        print(os.path.join(dataset_path, end_path))

        ecg_data = pd.read_pickle(os.path.join(dataset_path, end_path))

        if force_reprocess:
            ecg_data = process_data(ecg_data)
            ecg_data.to_pickle(os.path.join(dataset_path, f"ptECGs/filtered_{f_name}.pk"))

        return pt_data, ecg_data
    except (OSError, FileNotFoundError) as e:
        print(e)
        return


def load_feas_dataset(feas=2, save_name="dataframe", force_reload=False, process=True, force_reprocess=False, ecg_range=None, ecg_meas_diag=None, filter_func=None):
    dataset = None
    if not force_reload:
        dataset = load_feas_dataset_pickle(process, save_name, feas, force_reprocess)
    if dataset is None:
        print("Failed to load from pickle, regenerating files")
        dataset = load_feas_dataset_scratch(process, feas, ecg_range, ecg_meas_diag, save_name,  filter_func)
    else:
        dataset = filter_dataset(dataset[0], dataset[1], ecg_range, ecg_meas_diag)

    return dataset


def read_pt_record(x, dataset_path):
    try:
        record = wfdb.rdrecord(os.path.join(dataset_path, x['file_path']))
        n_ecgs = x["noRecs"]
        # print(f"found {record.p_signal.shape[1]} ECGs")
        return pd.DataFrame({'data': [record.p_signal[:, i] for i in range(n_ecgs)],
                             'adc_gain': [record.adc_gain[i] for i in range(n_ecgs)]})
    except (OSError, FileNotFoundError):
        return None


def process_ecgs_and_save_wfdb(feas, save_folder="ptECGs_processed", filter_undecided=False):
    dataset_path = dataset_paths[feas-1]

    save_path = os.path.join(dataset_path, save_folder)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    pt_data = load_pt_dataset(feas)
    ecg_data = load_ecg_csv(feas)

    if filter_undecided:
        ecg_data = ecg_data[ecg_data.measDiag != DiagEnum.Undecided]
        pt_data = pt_data[pt_data.ptID.isin(ecg_data.ptID)]

    print("reading ECG files")
    pt_data["processed_df"] = None
    ecg_data = ecg_data.drop(["data", "adc_gain", "length"], axis=1)

    pt_file_paths = pd.Series(index=pt_data.ptID, data=None)
    ecg_lens = pd.Series(index=ecg_data.index, data=None)

    for _, x in tqdm(pt_data.iterrows(), total=len(pt_data.index)):
        ecg_df = read_pt_record(x, dataset_path)
        if ecg_df is None:
            print(f"cannot read pt: {x.ptID}")
            continue

        ecg_df_pt = ecg_data[ecg_data.ptID == x.ptID].copy()
        ecg_df_pt.index = ecg_df_pt.measNo - 1

        ecg_df_processed = process_data(ecg_df.loc[ecg_df_pt.index], parallel=False)
        ecg_df_pt = pd.concat([ecg_df_pt, ecg_df_processed], axis=1, join="inner")

        folder = os.path.join(save_path, f"{str(int(math.floor(x.ptID/100.0)*100)).zfill(6)}")
        if not os.path.isdir(folder):
            os.mkdir(folder)
        fname = os.path.join(folder, f"saferF{feas}_pt{str(x.ptID).zfill(6)}.pk")
        ecg_df_pt.to_pickle(fname)

        # pt_data.loc[x.ptID, "processed_df"] = os.path.relpath(fname, dataset_path)
        pt_file_paths.loc[x.ptID] = os.path.relpath(fname, dataset_path)

        sig_lens = ecg_df_pt.data.map(lambda x: x.shape[0]).values
        ecg_lens.loc[ecg_data.loc[ecg_data.ptID == x.ptID].index] = sig_lens

    pt_data["processed_df"] = pt_file_paths
    ecg_data["length"] = ecg_lens

    pt_data.to_csv(os.path.join(dataset_path, "pt_data_anon_processed_paths.csv"))
    ecg_data.to_csv(os.path.join(dataset_path, "rec_data_anon_processed.csv"))


if __name__ == "__main__":

    """
    pt_data = pd.read_csv(os.path.join(feas1_path, "pt_data_anon_processed_paths.csv"))

    def add_filename(x):
        if not isinstance(x.processed_df, str):
            return None
        else:
            return os.path.join(x.processed_df, os.path.basename(x.file_path) + ".pk")
    
    def correct_filename(x):
        if not isinstance(x.processed_df, str):
            return None
        else:
            return os.path.split(x.processed_df)[0]

    pt_data["processed_df"] = pt_data.apply(correct_filename, axis=1)
    pt_data.to_csv(os.path.join(feas1_path, "pt_data_anon_processed_paths.csv"))
    """

    process_ecgs_and_save_wfdb(1, filter_undecided=True)


    """
    feas = 1

    if feas == 1 or feas == 3:
        # Load up in
        pt_data = load_pt_dataset(feas)
        num_chunks = math.ceil(pt_data.noRecs.sum()/chunk_size)

        for n in range(num_chunks):
            ecg_range = [chunk_size * n,
                         chunk_size * (n + 1)]

            load_feas_dataset(feas, ecg_range=ecg_range,
                              force_reload=True, force_reprocess=True,
                              save_name=f"dataframe_{n}")
    """


