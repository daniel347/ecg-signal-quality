import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_section1 = nn.Sequential(
            nn.Conv1d(1, 128, 17, stride=4, padding=8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128)
        )

        self.conv_section2 = nn.Sequential(
            nn.Conv1d(128, 256, 11, stride=2, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(256)
        )

        self.conv_section3 = nn.Sequential(
            nn.Conv1d(256, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(256)
        )

        self.conv_section4 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),
            nn.BatchNorm1d(128)
        )

        self.conv_section5 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),
            nn.BatchNorm1d(64)
        )

        self.conv_section6 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32)
        )

        self.lstm_n_hidden = 32
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, bidirectional=True, batch_first=True)

        # self.dense1 = nn.Linear(352, 128)
        self.dense2 = nn.Linear(64, 16)
        self.dense3 = nn.Linear(16, 1)

        self.activation = nn.ReLU()
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
        x = torch.flatten(torch.transpose(h, 0, 1), 1, 2)  # torch.flatten(x, 1, -1) #

        # [1152]
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # [128]
        x = self.dense3(x)
        # x = self.logsoftmax(x)
        # x = torch.nn.functional.sigmoid(x)

        # [1]
        return x