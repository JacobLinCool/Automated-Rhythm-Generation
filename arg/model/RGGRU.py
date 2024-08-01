import torch.nn as nn
from .BaseModel import BaseModel


class RGGRU(BaseModel):
    def __init__(self, hidden_dim=128, output_dim=3, num_layers=2):
        super().__init__()
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5),
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4),
        )

        # RNN for temporal dependency
        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        # x: (batch_size, seq_len, 1)
        batch_size, seq_len, _ = x.size()

        # Extract features with CNN
        x = self.cnn(x.view(batch_size, 1, seq_len))  # (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)

        # RNN for temporal dependencies
        x, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim*2)

        # Fully connected layer for classification
        x = self.fc(x)  # (batch_size, seq_len, output_dim)

        return x


if __name__ == "__main__":
    model = RGGRU()
    model.print_summary()
    model.dummy_test()
