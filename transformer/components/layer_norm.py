import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, model_dim: int, epsilon: float = 1e-9) -> None:
        """
        Layer normalization constructor

        Args:
            model_dim (int): Model dimensions
            epsilon (float, optional): Numerical stability in case the denominator
            becomes zero by chance. Defaults to 1e-9.
        """
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Layer normalization forward function

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Normalized tensor by layer
        """
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.epsilon)
        x_scaled_shifted = x_norm * self.gamma + self.beta

        return x_scaled_shifted
