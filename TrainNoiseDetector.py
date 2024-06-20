import pandas as pd
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import copy

from Models.NoiseCNN import CNN, hyperparameters

from Utilities.Training import *
from DataHandlers.Dataloaders import ECGDataset as Dataset

from Utilities.General import get_torch_device
from Utilities.Plotting import *

from DataHandlers.SAFERDatasetV2 import SaferDataset, feas2_path, trial_path
from DataHandlers.DiagEnum import DiagEnum
from DataHandlers.DataProcessUtilities import adaptive_gain_norm

import tqdm

def filter_ecgs_with_split(pt, ecg, pt_split=None):
    ecg_new = ecg[ecg.length == 9120]
    ecg_new = ecg_new[ecg_new.measDiag != DiagEnum.Undecided]
    ecg_new = ecg_new[ecg_new.measDiagAgree |
                      (ecg_new.measDiagRev1 == DiagEnum.Undecided) |
                      (ecg_new.measDiagRev2 == DiagEnum.Undecided)]
    pt_new = pt[pt.ptID.isin(ecg_new.ptID)]

    if pt_split is not None:
        # Select only participants from pt_split
        pt_new = pt_new[pt_new.ptID.isin(pt_split.index)]
        ecg_new = ecg_new[ecg_new.ptID.isin(pt_new.index)]

    return pt_new, ecg_new

def label_noise(x):
    return int(x == DiagEnum.PoorQuality)

def preprocess(x):
    return adaptive_gain_norm(x, 601)

if __name__ == "__main__":
    # ====  Options  ====
    enable_cuda = True
    out_model_name = "cnn_model_trial_trained_10_jun_2024_adaptive_gain"
    dataset_split_name = "9_Jun_2024_trial"

    batch_size = 32
    num_epochs = 50
    early_stop_num = 6
    # ========

    device = get_torch_device(enable_cuda)

    """
    train_dataset = pd.read_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_split_name}_train.pk"))
    test_dataset = pd.read_pickle(os.path.join(constants.feas2_path, f"ECGs/{dataset_split_name}_test.pk"))
    
    train_dataset = train_dataset[train_dataset["length"] == 9120]
    test_dataset = test_dataset[test_dataset["length"] == 9120]
    
    print(train_dataset["class_index"].value_counts())
    print(test_dataset["class_index"].value_counts())
    
    torch_dataset_train = Dataset(train_dataset)
    train_dataloader = DataLoader(torch_dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    torch_dataset_test = Dataset(test_dataset)
    test_dataloader = DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)
    """

    train_pts = pd.read_pickle(os.path.join(trial_path, f"splits/{dataset_split_name}_train_pts.pk"))
    test_pts = pd.read_pickle(os.path.join(trial_path, f"splits/{dataset_split_name}_test_pts.pk"))

    train_dataset = SaferDataset(feas=3,
                                 label_gen=label_noise,
                                 filter_func=lambda pt, ecg: filter_ecgs_with_split(pt, ecg, train_pts),
                                 preprocess_func=preprocess)

    test_dataset = SaferDataset(feas=3,
                                label_gen=label_noise,
                                filter_func=lambda pt, ecg: filter_ecgs_with_split(pt, ecg, test_pts),
                                preprocess_func=preprocess)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model = CNN(**hyperparameters).to(device)

    a = (len(train_dataset) - train_dataset.ecg_data.class_index.round().sum())/len(train_dataset)
    print(a)

    loss_func = binary_focal_loss(a, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 0.00002
    scheduler = StepLR(optimizer, 5, 0.5)  # 100, 1

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
        for i, (signals, labels, _) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            signals = torch.unsqueeze(signals.to(device), 1).float()

            labels = labels.float()

            optimizer.zero_grad()
            output = model(signals).to("cpu")[:, 0]
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Add the number of signals in this batch to a counter
            num_batches += signals.shape[0]
            total_loss += float(loss)

        print(f"Epoch {epoch} finished with average loss {total_loss / num_batches}")
        print("Testing ...")
        # Test
        num_test_batches = 0
        test_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (signals, labels, _) in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                signals = torch.unsqueeze(signals.to(device), 1).float()

                labels = labels.float()
                output = model(signals).to("cpu")[:, 0]
                loss = loss_func(output, labels)
                test_loss += float(loss)
                num_test_batches += signals.shape[0]

        print(f"Average test loss: {test_loss / num_test_batches}")
        losses.append([total_loss / num_batches, test_loss / num_test_batches])

        if test_loss / num_test_batches < best_test_loss:
            best_model = copy.deepcopy(model).cpu()
            best_test_loss = test_loss / num_test_batches
            best_epoch = epoch
        else:
            if best_epoch + early_stop_num <= epoch:
                break
    scheduler.step()

    model = best_model.to(device)

    losses_np = np.array(losses)
    np.save(f"TrainedModels/{out_model_name}_losses", losses_np)
    torch.save(model.state_dict(), f"TrainedModels/{out_model_name}.pt")
