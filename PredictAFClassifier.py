import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from Models.SpectrogramTransformerAttentionPooling import TransformerModel

from Utilities.Predict import *
from Utilities.General import get_torch_device

from DataHandlers.Dataloaders import RRiECGDataset as Dataset
from DataHandlers.SAFERDataset import feas1_path, feas2_path, num_chunks, chunk_size
from DataHandlers.CinC2020Dataset import cinc_2020_path
from DataHandlers.CinCDataset import cinc_2017_path


# ====  Options  ====
enable_cuda = True
model_name = "Transformer_20_may_cinc_train_attention_pooling_augmentation_smoothing"
dataset_path = os.path.join(feas1_path, "ECGs/feas1_27_mar_val.pk")
data_is_safer_pt = False  # True if dataset_split_name contains patients from safer rather than ECGs
# =======


device = get_torch_device(enable_cuda)

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

model.eval()

true_labels = []
predictions = []

outputs = []
inds = []

with torch.no_grad():
    for i, (signals, labels, ind) in enumerate(val_dataloader):
        signal = signals[0].to(device).float()
        rris = signals[1].to(device).float()
        rri_len = signals[2].to(device).float()

        labels = labels.long().detach().numpy()
        true_labels.append(labels)

        output = model(signal, rris, rri_len).detach().to("cpu").numpy()

        prediction = output
        predictions.append(prediction)

        for i, o in zip(ind, output):
            outputs.append(o)
            if isinstance(i, str):
                inds.append(i)
            else:
                inds.append(i.item())

dataset["prediction"] = pd.Series(data=outputs, index=inds)

predictions = np.concatenate(predictions)
true_labels = np.concatenate(true_labels)

conf_mat = confusion_matrix(true_labels, np.argmax(predictions, axis=1))
print_af_results(conf_mat)






