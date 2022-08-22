import torch
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding describes the location or position of an entity in a
    sequence so that each position is assigned a unique representation.
    """

    def __init__(self, max_seq_len: int, model_dim: int, n=10000) -> None:
        """
        Constructor of positional encoding.

        Args:
            max_seq_len (int): Maximum sequence length
            model_dim (int): Model dimension
            n (int, optional): User defined scalar. Defaults to 10000.
        """
        super().__init__()

        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        i2 = torch.arange(0, model_dim, step=2).float()

        self.encoding = torch.zeros(max_seq_len, model_dim)
        self.encoding[:, 0::2] = torch.sin(position / (n ** (i2 / model_dim)))
        self.encoding[:, 1::2] = torch.cos(position / (n ** (i2 / model_dim)))
        self.encoding.requires_grad = False

    def forward(self, x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]


def get_positional_encoding(seq_len: int, model_dim: int, n=10000) -> np.ndarray:
    """
    Generates positional encodings. Used only for testing purposes!

    Implemented using single loops for testing and easier understanding from the
    math provided.

    Args:
        seq_len (int): Sequence length
        model_dim (int): Model dimension
        n (int, optional): User defined scalar. Defaults to 10000.

    Returns:
        np.ndarray: Positional encoding
    """
    P = np.zeros((seq_len, model_dim))
    for k in range(seq_len):
        for i in range(model_dim // 2):
            x = k / np.power(n, ((2 * i) / model_dim))
            P[k, 2 * i] = np.sin(x)
            P[k, (2 * i) + 1] = np.cos(x)

    return P
