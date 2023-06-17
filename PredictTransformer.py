import sys

from DataHandlers.DiagEnum import DiagEnum
import DataHandlers.CinC2020Dataset as CinC2020Dataset
import DataHandlers.CinC2020Enums
import DataHandlers.SAFERDataset as SAFERDataset
import DataHandlers.CinCDataset as CinCDataset
import DataHandlers.MITNSTDataset as MITNSTDataset

from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset, DataLoader

from Models.SpectrogramTransformerAttentionPooling import TransformerModel
from torch.optim.lr_scheduler import LambdaLR

from Utilities.Training import *

import torch
from sklearn.model_selection import train_test_split
from Utilities.Predict import *
import Utilities.constants as constants
import os

from DataHandlers.Dataloaders import RRiECGDataset as Dataset

# A fudge because I moved the files
sys.modules["SAFERDataset"] = SAFERDataset
sys.modules["CinC2020Dataset"] = CinC2020Dataset
sys.modules["DiagEnum"] = DataHandlers.DiagEnum
sys.modules["CinC2020Enums"] = DataHandlers.CinC2020Enums
sys.modules["CinCDataset"] = CinCDataset


# ====  Options  ====
enable_cuda = True
model_name = "Transformer_16_June_safer_train_attention_pooling_augmentation_smoothing"
dataset_path = os.path.join(constants.feas1_path, "ECGs/feas1_27_mar_val.pk" )
data_is_safer_pt = False  # True if dataset_split_name contains patients from safer rather than ECGs

# =======


if torch.cuda.is_available() and enable_cuda:
    print("Using Cuda")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

if data_is_safer_pt:
    dataset_pts = pd.read_pickle(dataset_path)
    # We only use patient lists for the entirety of feas1 because its too big
    # See training code using DatasetSequenceIterator to use this
    raise NotImplementedError

else:
    dataset = pd.read_pickle(dataset_path)
    print(dataset.head())


torch_dataset = Dataset(dataset)
val_dataloader = DataLoader(torch_dataset, batch_size=128, shuffle=True, pin_memory=True)

n_head = 4
n_fft = 128
embed_dim = 128
n_inp_rri = 64

model = TransformerModel(3, embed_dim, n_head, 512, 6, n_fft, n_inp_rri, device=device).to(device)
model = model.to(device)
model.load_state_dict(torch.load(f"TrainedModels/{model_name}.pt", map_location=device))

predictions, true_labels = get_predictions_transformer(model, val_dataloader, dataset, device)

conf_mat = confusion_matrix(true_labels, np.argmax(predictions, axis=1))
print_results(conf_mat)






