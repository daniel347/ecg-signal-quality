import importlib
from Models.NoiseCNN import CNN, hyperparameters
import pandas as pd
import torch
from torch.utils.data import DataLoader
from DataHandlers.Dataloaders import RRiECGDataset as Dataset
from DataHandlers.Dataloaders import DatasetSequenceIterator

# ====  Options  ====
enable_cuda = True
model_name = "CNN_16_may_final_no_undecided"
dataset_split_name = "19_May_cinc_2017"
data_is_safer_pt = False  # True if dataset_split_name contains patients from safer

output_name = f"{dataset_split_name}_noise_predictions"

batch_size = 128

# =======

if torch.cuda.is_available() and enable_cuda:
    print("Using Cuda")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

dataset = pd.read_pickle(f"TrainedModels/{dataset_split_name}_train.pk")
torch_dataset = Dataset(dataset)
dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

noiseDetector = CNN(**hyperparameters).to(device)
noiseDetector.load_state_dict(torch.load(f"TrainedModels/{model_name}.pt", map_location=device))
noiseDetector.eval()

inds = []
noise_ps = []

with torch.no_grad():
    for i, (signals, labels, ind) in enumerate(dataloader):
        signal = signals[0].to(device).float()
        noise_prob = noiseDetector(torch.unsqueeze(signal, 1)).detach().to("cpu").numpy()

        for i, n in zip(ind, noise_prob):
            if type(i) == str:
                inds.append(i)
            else:
                inds.append(i.item())
            noise_ps.append(float(n))

noise_predictions = pd.Series(data=noise_ps, index=inds)
noise_predictions.to_pickle(f"TrainedModels/{output_name}.pk")