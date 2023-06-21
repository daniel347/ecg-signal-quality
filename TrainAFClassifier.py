import pandas as pd
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

import DataHandlers.MITNSTDataset as MITNSTDataset
from DataHandlers.Dataloaders import RRiECGDataset as Dataset
from DataHandlers.Dataloaders import DatasetSequenceIterator, load_feas1_chunk_range, prepare_safer_data
from DataHandlers.SAFERDataset import feas1_path, feas2_path, num_chunks, chunk_size
from DataHandlers.CinC2020Dataset import cinc_2020_path
from DataHandlers.CinCDataset import cinc_2017_path

from Models.SpectrogramTransformerAttentionPooling import TransformerModel

from Utilities.Training import *
from Utilities.General import get_torch_device


# ====  Options  ====
enable_cuda = True
out_model_name = "transformer_model"

pre_trained_model_name = "Transformer_20_may_cinc_train_attention_pooling_augmentation_smoothing"
dataset_split_name = os.path.join(feas1_path, "ECGs/feas1_27_mar")
dataset_type = "safer_feas1"

noise_predictions_name = os.path.join(feas1_path, "18_Jun_feas1_test_train_pts_noise_predictions")
p_noisy_threshold = 1  # Only allow signals below this probability of noise to be used in training and testing

augment_noise = False

batch_size = 64
num_epochs = 1
early_stop_num = 5
# =======

device = get_torch_device(enable_cuda)

# Load the dataset splits
if dataset_type == "safer_feas1":
    train_pts = pd.read_pickle(f"{dataset_split_name}_train_pts.pk")

    def load_feas1_nth_chuck(n):
        ecg_data, pt_data = load_feas1_chunk_range((n, n + 1))
        ecg_data.index = ecg_data["measID"]
        pt_data.index = pt_data["ptID"]
        return prepare_safer_data(pt_data, ecg_data)[1]

    if p_noisy_threshold < 1:
        noise_pred_df = pd.read_pickle(f"{noise_predictions_name}.pk")

        # Create some a filter function to select data from each partition
        def filter_train_pts(ecg_data):
            return ecg_data[ecg_data["ptID"].isin(train_pts["ptID"]) & (noise_pred_df <= p_noisy_threshold)]
    else:
        def filter_train_pts(ecg_data):
            return ecg_data[ecg_data["ptID"].isin(train_pts["ptID"])]



    loading_functions = [lambda n=n: load_feas1_nth_chuck(n) for n in
                         range(num_chunks)]

    train_dataloader = DatasetSequenceIterator(loading_functions,
                                                [batch_size for n in
                                                range(num_chunks)],
                                                filter=filter_train_pts)

    # Load the test dataset directly from file
    feas1_test_dataset = pd.read_pickle(f"{dataset_split_name}_test.pk")
    if p_noisy_threshold < 1:
        feas1_test_dataset = feas1_test_dataset[noise_pred_df <= p_noisy_threshold]
    torch_dataset_test = Dataset(feas1_test_dataset)
    test_dataloader = DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)
else:
    train_dataset = pd.read_pickle(f"{dataset_split_name}_train.pk")
    test_dataset = pd.read_pickle(f"{dataset_split_name}_test.pk")

    if p_noisy_threshold < 1:
        noise_pred_df = pd.read_pickle(f"{noise_predictions_name}_train.pk")
        train_dataset = train_dataset[noise_pred_df <= p_noisy_threshold]
        test_dataset = test_dataset[noise_pred_df <= p_noisy_threshold]

    torch_dataset_test = Dataset(test_dataset)
    test_dataloader = DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)

    torch_dataset_train = Dataset(train_dataset)

    if augment_noise:
        # We load the MIT NST dataset to obtain the noise signals for data augmentation
        noise_df = MITNSTDataset.load_noise_segments(["em", "ma"], 3000)
        torch_dataset_train.set_noise_prob(0.1, 0.2, noise_df)

    train_dataloader = DataLoader(torch_dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)

# Initialise the model
n_head = 4
n_fft = 128
embed_dim = 128
n_inp_rri = 64

model = TransformerModel(3, embed_dim, n_head, 512, 6, n_fft, n_inp_rri, device=device).to(device)
if pre_trained_model_name is not None:
    model.load_state_dict(torch.load(f"TrainedModels/{pre_trained_model_name}.pt", map_location=device))

# Set up training
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

# Perform the training
best_test_loss = 100
best_epoch = -1
best_model = copy.deepcopy(model).cpu()

losses = []

for epoch in range(num_epochs):
    total_loss = 0
    print(f"starting epoch {epoch} ...")
    # Train
    num_batches = 0
    model.train()
    for i, (signals, labels, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        signal = signals[0].to(device).float()
        rris = signals[1].to(device).float()
        rri_len = signals[2].to(device).float()

        if torch.any(torch.isnan(signal)) or torch.any(torch.isnan(rris)):
            print("Signals are nan")
            continue

        labels = labels.long()
        optimizer.zero_grad()
        output = model(signal, rris, rri_len).to("cpu")

        if torch.any(torch.isnan(output)):
            print("output is nan")
            raise ValueError

        loss = loss_func(output, labels)
        if torch.isnan(loss):
            print("loss is nan")
            raise ValueError

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        scheduler.step()
        num_batches += 1
        total_loss += float(loss)

    print(f"Epoch {epoch} finished with average loss {total_loss/num_batches}")
    print("Testing ...")

    num_test_batches = 0
    test_loss = 0
    with torch.no_grad():
        model.eval()
        for i, (signals, labels, _) in enumerate(test_dataloader):
            signal = signals[0].to(device).float()
            rris = signals[1].to(device).float()
            rri_len = signals[2].to(device).float()

            if torch.any(torch.isnan(signal)) or torch.any(torch.isnan(rris)):
                print("Signals are nan")
                continue

            labels = labels.long()
            output = model(signal, rris, rri_len).to("cpu")
            loss = loss_func(output, labels)
            test_loss += float(loss)
            num_test_batches += 1

    print(f"Average test loss: {test_loss/num_test_batches}")
    losses.append([total_loss/num_batches, test_loss/num_test_batches])

    if test_loss/num_test_batches < best_test_loss:
        best_model = copy.deepcopy(model).cpu()
        best_test_loss = test_loss/num_test_batches
        best_epoch = epoch
    else:
        if best_epoch + early_stop_num <= epoch:
            break

model = best_model.to(device)

losses_np = np.array(losses)
np.save(f"TrainedModels/{out_model_name}_losses", losses_np)
torch.save(model.state_dict(), f"TrainedModels/{out_model_name}.pt")