import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.
    """

    def __init__(self) -> None:
        """
        Scaled Dot-Product Attention constructor.
        """
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Scaled Dot-Product Attention forward function.

        Args:
            query (torch.Tensor): Query vector
            key (torch.Tensor): Key vector
            value (torch.Tensor): Value vector

        Returns:
            torch.Tensor: Scaled Dot-Product Attention scores
        """
        assert (
            len(key.shape) == 4 or len(query.shape) == 4 or len(value.shape) == 4
        ), "Key, Query and Values should be 4D tensors"

        kt = torch.transpose(key, 2, 3)
        qk = torch.matmul(query, kt)

        kd = key.shape[-1]
        qk_den = np.sqrt(kd)

        qk_scaled = qk / qk_den

        qk_norm = self.softmax(qk_scaled)

        qkv = torch.matmul(qk_norm, value)

        return qkv
