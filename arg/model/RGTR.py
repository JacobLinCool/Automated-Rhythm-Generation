import torch
import torch.nn as nn
from .BaseModel import BaseModel


class RGTransformer(BaseModel):
    def __init__(
        self,
        hidden_dim=128,
        output_dim=3,
        num_layers=2,
        nhead=8,
        dim_feedforward=512,
        max_audio_seconds=600,
    ):
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
                in_channels=32,
                out_channels=hidden_dim,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4),
        )

        # Positional encoding for transformer
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 200 * max_audio_seconds, hidden_dim)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, 1)
        batch_size, seq_len, _ = x.size()
        # print(f"Input shape: {x.shape}")

        # Extract features with CNN
        x = self.cnn(x.view(batch_size, 1, seq_len))  # (batch_size, channels, seq_len)
        # print(f"After CNN shape: {x.shape}")
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        # print(f"After permute shape: {x.shape}")

        # Add positional encoding
        pos_enc = self.positional_encoding[:, : x.size(1), :]
        # print(f"Positional encoding shape: {pos_enc.shape}")
        x = x + pos_enc
        # print(f"After adding positional encoding shape: {x.shape}")

        # Transformer for temporal dependencies
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        # print(f"After transformer shape: {x.shape}")

        # Fully connected layer for classification
        x = self.fc(x)  # (batch_size, seq_len, output_dim)
        # print(f"Output shape: {x.shape}")

        return x


if __name__ == "__main__":
    model = RGTransformer()
    model.print_summary()
    model.dummy_test()
