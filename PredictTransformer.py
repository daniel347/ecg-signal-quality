import sys

from DataHandlers.DiagEnum import DiagEnum
import DataHandlers.CinC2020Dataset as CinC2020Dataset
import DataHandlers.CinC2020Enums
import DataHandlers.MITNSTDataset as MITNSTDataset

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
enable_cuda = True
model_name = "Transformer_24_May_cinc_2017_train_attention_pooling_augmentation_smoothing"
dataset_split_name = "19_May_cinc_2017"
data_is_safer_pt = False  # True if dataset_split_name contains patients from safer

# =======

def load_feas1_chunk_range(chunk_range=(0, num_chunks)):
    ecg_data = []
    pt_data = []

    for chunk_num in range(chunk_range[0], chunk_range[1]):
        feas1_pt_data, feas1_ecg_data = SAFERDataset.load_feas_dataset(1, f"dataframe_{chunk_num}.pk")

        ecg_data.append(feas1_ecg_data)
        pt_data.append(feas1_pt_data)

    feas1_ecg_data = pd.concat(ecg_data)
    feas1_ecg_data["feas"] = 1
    feas1_ecg_data["rri_len"] = feas1_ecg_data["rri_feature"].map(lambda x: x[x > 0].shape[-1])
    feas1_pt_data = pd.concat(pt_data).drop_duplicates()

    return feas1_ecg_data, feas1_pt_data

if torch.cuda.is_available() and enable_cuda:
    print("Using Cuda")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

if data_is_safer_pt:

else:
    val_dataset = pd.read_pickle(f"TrainedModels/{dataset_split_name}_val.pk")


torch_dataset_val = Dataset(val_dataset)
val_dataloader = DataLoader(torch_dataset_val, batch_size=128, shuffle=True, pin_memory=True)

n_head = 4
n_fft = 128
embed_dim = 128
n_inp_rri = 64

model = TransformerModel(3, embed_dim, n_head, 512, 6, n_fft, n_inp_rri, device=device).to(device)
model = model.to(device)
model.load_state_dict(torch.load(f"TrainedModels/{model_name}.pt", map_location=device))

predictions, true_labels = get_predictions_transformer(model, val_dataloader, val_dataset, device)




