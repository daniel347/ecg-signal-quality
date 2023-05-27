import torch
import torch.nn as nn
import math

# Now define a model
hyperparameters = {"lr": 0.00075, "k1": 17, "k2": 11, "k3": 5, "k4": 3, "k5": 3, "k6": 3,
                   "c1": 128, "c2": 256, "c3": 256, "c4": 128, "c5": 64, "c6": 32,
                   "lstm_n_hidden": 32, "dense1": 256, "dense2": 16, "gamma": 0.5, "sched_gamma": 0.5, "sched_step": 6}

class CNN(nn.Module):

    def __init__(self, k1, k2, k3, k4, k5, k6, c1, c2, c3, c4, c5, c6, lstm_n_hidden, dense2, **_):
        super(CNN, self).__init__()

        self.conv_section1 = nn.Sequential(
            nn.Conv1d(1, c1, k1, stride=4, padding=math.ceil((k1 - 4) / 2)),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(c1)
        )

        self.conv_section2 = nn.Sequential(
            nn.Conv1d(c1, c2, k2, stride=2, padding=math.ceil((k1 - 2) / 2)),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(c2)
        )

        self.conv_section3 = nn.Sequential(
            nn.Conv1d(c2, c3, k3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(c3)
        )

        self.conv_section4 = nn.Sequential(
            nn.Conv1d(c3, c4, k4, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),
            nn.BatchNorm1d(c4)
        )

        self.conv_section5 = nn.Sequential(
            nn.Conv1d(c4, c5, k5, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),
            nn.BatchNorm1d(c5)
        )

        self.conv_section6 = nn.Sequential(
            nn.Conv1d(c5, c6, k6, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(c6)
        )

        self.lstm_n_hidden = lstm_n_hidden
        self.lstm = nn.LSTM(input_size=c6, hidden_size=lstm_n_hidden, bidirectional=True, batch_first=True)

        self.dense2 = nn.Linear(2 * lstm_n_hidden, dense2)
        self.dense3 = nn.Linear(dense2, 1)

        self.activation = nn.ELU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout()

    def init_lstm_hidden(self, batch_size, device):
        # This resets the LSTM hidden state after each batch
        hidden_state = torch.zeros(2, batch_size, self.lstm_n_hidden, device=device)
        cell_state = torch.zeros(2, batch_size, self.lstm_n_hidden, device=device)
        return (hidden_state, cell_state)

    def forward(self, x):
        # [1, 9120]
        x = self.conv_section1(x)

        # [512, 1140]
        x = self.conv_section2(x)

        # [256, 570]
        x = self.conv_section3(x)

        # [128, 285]
        x = self.conv_section4(x)

        # [64, 143]
        x = self.conv_section5(x)

        # [32, 72]
        x = self.conv_section6(x)

        # [32, 36]
        x = torch.transpose(x, 1, 2)

        x, (h, _) = self.lstm(x, self.init_lstm_hidden(x.shape[0], x.device))
        x = torch.flatten(torch.transpose(h, 0, 1), 1, 2)  # torch.flatten(x, 1, -1)
        # x = torch.flatten(x, 1, -1)

        # [1152]
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # [128]
        x = self.dense3(x)
        # x = self.logsoftmax(x)

        # [1]
        return x