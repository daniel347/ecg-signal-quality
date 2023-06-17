import sys
import pandas as pd
import os

from DataHandlers.DiagEnum import DiagEnum
import DataHandlers.CinC2020Dataset as CinC2020Dataset
import DataHandlers.CinC2020Enums
import DataHandlers.SAFERDataset as SAFERDataset
import DataHandlers.MITNSTDataset as MITNSTDataset

from Models.SpectrogramTransformerAttentionPooling import TransformerModel
from torch.optim.lr_scheduler import LambdaLR

from Utilities.Training import *
from DataHandlers.Dataloaders import RRiECGDataset as Dataset
from DataHandlers.Dataloaders import DatasetSequenceIterator

import Utilities.constants as constants

from DataHandlers.Dataloaders import load_feas1_chunk_range, prepare_safer_data

import torch

# A fudge because I moved the files
sys.modules["CinC2020Dataset"] = CinC2020Dataset
sys.modules["DiagEnum"] = DataHandlers.DiagEnum
sys.modules["CinC2020Enums"] = DataHandlers.CinC2020Enums


# ====  Options  ====
enable_cuda = True
out_model_name = "Transformer_16_June_safer_train_attention_pooling_augmentation_smoothing"

pre_trained_model_name = "Transformer_16_June_cinc_2020_train_attention_pooling_augmentation_smoothing"
dataset_split_name = "feas1_27_mar"
dataset_type = "safer_feas1"

augment_noise = False
remove_noisy_samples = False

batch_size = 64

# =======

if torch.cuda.is_available() and enable_cuda:
    print("Using Cuda")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# Load the dataset splits
if dataset_type == "safer_feas1":
    train_pts = pd.read_pickle(os.path.join(constants.feas1_path, f"ECGs/{dataset_split_name}_train_pts.pk"))

    # Create some a filter function to select data from each partition
    def filter_train_pts(ecg_data):
        return ecg_data[ecg_data["ptID"].isin(train_pts["ptID"])]

    def load_feas1_nth_chuck(n):
        ecg_data, pt_data = load_feas1_chunk_range((n, n + 1))
        ecg_data.index = ecg_data["measID"]
        pt_data.index = pt_data["ptID"]
        return prepare_safer_data(pt_data, ecg_data)[1]


    loading_functions = [lambda n=n: load_feas1_nth_chuck(n) for n in
                         range(constants.num_chunks)]

    train_dataloader = DatasetSequenceIterator(loading_functions,
                                                [batch_size for n in
                                                range(constants.num_chunks)],
                                                filter=filter_train_pts)

    # Load the test dataset directly from file
    feas1_test_dataset = pd.read_pickle(os.path.join(constants.feas1_path, f"ECGs/{dataset_split_name}_test.pk"))
    torch_dataset_test = Dataset(feas1_test_dataset)
    test_dataloader = DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)
else:
    train_dataset = pd.read_pickle(f"TrainedModels/{dataset_split_name}_train.pk")
    test_dataset = pd.read_pickle(f"TrainedModels/{dataset_split_name}_test.pk")

    torch_dataset_test = Dataset(test_dataset)
    test_dataloader = DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)

    torch_dataset_train = Dataset(train_dataset)

    if augment_noise:
        # We load the MIT NST dataset to obtain the noise signals for data augmentation
        noise_df = MITNSTDataset.load_noise_segments(["em", "ma"], 3000)
        torch_dataset_train.set_noise_prob(0.1, 0.2, noise_df)

    train_dataloader = DataLoader(torch_dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)


n_head = 4
n_fft = 128
embed_dim = 128
n_inp_rri = 64

model = TransformerModel(3, embed_dim, n_head, 512, 6, n_fft, n_inp_rri, device=device).to(device)

if dataset_type != "safer_feas1":
    class_counts = torch.tensor(train_dataset["class_index"].value_counts().sort_index().values.astype(np.float32))
    class_weights = (1/class_counts)
    class_weights /= torch.sum(class_weights)
else:
    class_counts = torch.tensor([train_pts["noNormalRecs"].sum(),
                                 train_pts["noAFRecs"].sum(),
                                 train_pts["noOtherRecs"].sum()])

    class_weights = (1/class_counts)
    class_weights /= torch.sum(class_weights)


loss_func = focal_loss(class_weights, 2, 0) # 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)  # 0.0005

number_warmup_batches = 600
scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda(number_warmup_batches))

model = model.to(device)
model.fix_transformer_params(fix_spec=False, fix_rri=False)
num_epochs = 1

model, losses = train_transformer(model, device, train_dataloader,
                                  test_dataloader, optimizer, loss_func,
                                  scheduler, num_epochs, 5)
model = model.to(device)

losses_np = np.array(losses)
np.save(f"TrainedModels/{out_model_name}", losses_np)
torch.save(model.state_dict(), f"TrainedModels/{out_model_name}.pt")