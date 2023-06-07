import torch
import torch.nn as nn
import math

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_section1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )

        self.conv_section2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )

        self.conv_section3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128)
        )

        self.conv_section4 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )

        self.conv_section5 = nn.Sequential(
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )

        self.lstm_n_hidden = 32
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, bidirectional=True, batch_first=True)

        # self.dense1 = nn.Linear(352, 128)
        self.dense2 = nn.Linear(896, 256)
        self.dense3 = nn.Linear(256, 1)

        self.activation = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout()

    def init_lstm_hidden(self, batch_size, device):
        # This resets the LSTM hidden state after each batch
        hidden_state = torch.zeros(2, batch_size, self.lstm_n_hidden, device=device)
        cell_state = torch.zeros(2, batch_size, self.lstm_n_hidden, device=device)
        return (hidden_state, cell_state)

    def forward(self, x):

        # [batch, 1, 40, 141]
        x = self.conv_section1(x)

        # [batch, 32, 38, 139]
        x = self.conv_section2(x)

        # [batch, 64, 18, 68]
        x = self.conv_section3(x)

        # [batch, 128, 7, 32]
        x = self.conv_section4(x)

        # [batch, 64, 5, 30]
        x = self.conv_section5(x)

        # [batch, 32, 1, 14]
        x = x[:, :, 0, :]
        x = torch.transpose(x, 1, 2)

        x, _ = self.lstm(x, self.init_lstm_hidden(x.shape[0], x.device))
        x = torch.flatten(x, 1, -1)

        # [batch, 896]
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # [batch, 256]
        x = self.dense3(x)
        # x = self.logsoftmax(x)

        # [4]
        return x