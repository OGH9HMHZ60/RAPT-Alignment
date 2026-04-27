"""Transformer encoder for symbolic music alignment.

Defines the windowed Transformer encoder used to map pianoroll context windows
into a shared embedding space, together with a depthwise-convolutional
relative positional encoding suitable for unaligned symbolic sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPositionalEncoding(nn.Module):
    """Relative positional encoding via a depthwise 1D convolution.

    The convolution is applied along the temporal axis and added back to the
    input as a residual, providing translation-invariant positional context
    without relying on absolute positional embeddings.
    """

    def __init__(self, d_model: int, kernel_size: int = 31):
        """
        Args:
            d_model: Hidden embedding dimensionality.
            kernel_size: Receptive field of the depthwise convolution.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + x_conv


class PianoTransformer(nn.Module):
    """Transformer encoder mapping a pianoroll window to an L2-normalized embedding.

    A single forward pass consumes a fixed-size pianoroll window and returns one
    embedding vector per window, obtained by pooling the encoded sequence
    according to the configured pooling strategy. The encoder is used as a
    shared (Siamese) branch for both score and performance windows.
    """

    def __init__(
        self,
        in_dim: int = 89,
        d_model: int = 256,
        nhead: int = 4,
        nlayers: int = 3,
        ff: int = 512,
        pooling: str = "center",
        **kwargs,
    ):
        """
        Args:
            in_dim: Input feature dimensionality (88 piano keys + 1 onset channel).
            d_model: Hidden embedding dimensionality.
            nhead: Number of attention heads.
            nlayers: Number of stacked Transformer encoder layers.
            ff: Feed-forward dimensionality inside each encoder layer.
            pooling: Pooling strategy applied to the encoded window.
                One of {"center", "mean", "max"}.
        """
        super().__init__()
        self.pooling = pooling

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = ConvPositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, window_size, in_dim).

        Returns:
            L2-normalized embeddings of shape (batch, d_model).
        """
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.transformer(x)

        if self.pooling == "center":
            z = x[:, x.size(1) // 2, :]
        elif self.pooling == "mean":
            z = torch.mean(x, dim=1)
        elif self.pooling == "max":
            z = torch.max(x, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        z = self.norm(z)
        return F.normalize(z, dim=-1)
