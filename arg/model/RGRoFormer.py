import torch
import torch.nn as nn
from .BaseModel import BaseModel


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Create a positional embedding matrix with sinusoidal patterns.
        self.freqs = self._get_freqs()

    def _get_freqs(self):
        # Use frequencies from 1 to d_model / 2.
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model)
        )
        # Get a list of cosines and sines with increasing frequency.
        position = torch.arange(0, self.max_seq_len).float()
        freqs = torch.einsum("i,j->ij", position, inv_freq)
        return torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.size()
        freqs = (
            self.freqs[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        )  # Match the batch size and seq_len
        freqs = freqs.to(x.device)

        # Separate input tensor into two parts for real and imaginary components
        real_part, imag_part = x[..., : self.d_model // 2], x[..., self.d_model // 2 :]

        # Apply rotation
        x_rotated = torch.cat(
            (
                real_part * freqs[..., : self.d_model // 2]
                - imag_part * freqs[..., self.d_model // 2 :],
                real_part * freqs[..., self.d_model // 2 :]
                + imag_part * freqs[..., : self.d_model // 2],
            ),
            dim=-1,
        )

        return x_rotated


class RGRoFormer(BaseModel):
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
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead

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

        # Rotary Positional Encoding
        self.rotary_pos_encoding = RotaryPositionalEmbedding(
            hidden_dim, max_audio_seconds * 200
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
        x = self.rotary_pos_encoding(x)
        # print(f"After applying RoPE shape: {x.shape}")

        # Transformer for temporal dependencies
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        # print(f"After transformer shape: {x.shape}")

        # Fully connected layer for classification
        x = self.fc(x)  # (batch_size, seq_len, output_dim)
        # print(f"Output shape: {x.shape}")

        return x


if __name__ == "__main__":
    model = RGRoFormer()
    model.print_summary()
    model.dummy_test()
