import torch
from torch import nn


class PositionalFeedForward(nn.Module):
    def __init__(self, model_dim: int, hidden_size: int) -> None:
        """
        Positional Feed Forward constructor

        Args:
            model_dim (int): Model dimensions
            hidden_size (int): Hidden layer dimension
        """
        super().__init__()

        self.linear1 = nn.Linear(model_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, model_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Positional Feed Forward forward function

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Linearly transformed output
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
