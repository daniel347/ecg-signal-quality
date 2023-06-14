import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import DataHandlers.DataAugmentations as DataAugmentations
import numpy as np
from DataHandlers.DataProcessUtilities import *
import copy
from tqdm import tqdm

def get_lr_lambda(number_warmup_batches):
    def warmup(current_step: int):
        if current_step < number_warmup_batches:
            # print(current_step / number_warmup_batches ** 1.5)
            return current_step / number_warmup_batches ** 1.5
        else:
            # print(1/math.sqrt(current_step))
            return 1/math.sqrt(current_step)

    return warmup

class focal_loss(nn.Module):
    def __init__(self, weights, gamma=2, label_smoothing=0):
        super(focal_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        self.weights = weights
        self.gamma = gamma

    def forward(self, pred, targets):
        ce = self.ce_loss(pred, targets)
        pt = torch.exp(-ce)

        loss_sum = torch.sum(((1-pt) ** self.gamma) * ce * self.weights[targets])
        norm_factor = torch.sum(self.weights[targets])
        return loss_sum/norm_factor


def train_transformer(model, device, train_dataloader, test_dataloader, optimizer, loss_func, scheduler, num_epochs=40, early_stop_num=5):
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

            if torch.any(torch.isnan(signal)):
                print("Signals are nan")
                continue

            if torch.any(torch.isnan(rris)):
                print("Signals are nan")
                continue

            labels = labels.long()
            optimizer.zero_grad()
            output = model(signal, rris, rri_len).to("cpu")

            if torch.any(torch.isnan(output)):
                print(signal)
                print(rris)
                print(rri_len)
                print(output)
                raise ValueError

            loss = loss_func(output, labels)
            if torch.isnan(loss):
                raise ValueError
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            scheduler.step()
            num_batches += 1
            total_loss += float(loss)

        print(num_batches)

        print(f"Epoch {epoch} finished with average loss {total_loss/num_batches}")
        # writer.add_scalar("Loss/train", total_loss/num_batches, epoch)
        print("Testing ...")
        # Test
        num_test_batches = 0
        test_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (signals, labels, _) in enumerate(test_dataloader):
                signal = signals[0].to(device).float()
                rris = signals[1].to(device).float()
                rri_len = signals[2].to(device).float()

                if torch.any(torch.isnan(signal)):
                    print("Signals are nan")
                    continue

                labels = labels.long()
                output = model(signal, rris, rri_len).to("cpu")
                loss = loss_func(output, labels)
                test_loss += float(loss)
                num_test_batches += 1

        print(f"Average test loss: {test_loss/num_test_batches}")
        losses.append([total_loss/num_batches, test_loss/num_test_batches])
        # writer.add_scalar("Loss/test", test_loss/num_t est_batches, epoch)

        if test_loss/num_test_batches < best_test_loss:
            best_model = copy.deepcopy(model).cpu()
            best_test_loss = test_loss/num_test_batches
            best_epoch = epoch
        else:
            if best_epoch + early_stop_num <= epoch:
                return best_model, losses

    return best_model, losses