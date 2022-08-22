import torch
from torch import nn

from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head_attention import MultiHeadAttention
from transformer.components.positional_feed_forward import PositionalFeedForward
from transformer.embedding.transformer import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        dict_size: int,
        model_dim: int,
        max_seq_len: int,
        head_num: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        """
        Encoder constructor

        Args:
            dict_size (int): Size of the dictionary of embeddings
            model_dim (int): Model dimensions
            max_seq_len (int): Maximum sequence length
            head_num (int): Head number
            hidden_size (int): Hidden layer dimension
            num_layers (int): Number of layers
        """
        super().__init__()

        self.embedding = TransformerEmbedding(dict_size, model_dim, max_seq_len)
        layers = [
            EncoderLayer(model_dim, head_num, hidden_size) for i in range(num_layers)
        ]
        self.encoders = SequentialWrapper(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder forward function

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoder activations
        """
        act = self.embedding(x=x)
        act = self.encoders(x=act)

        return act


class SequentialWrapper(nn.Sequential):
    """
    Wrapper for nn.Sequential.

    Overrides default forward function in order to
    support custom arguments.
    """

    def forward(self, x: torch.Tensor):
        """
        Sequential wrapper forward override function

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output activations
        """
        act = x
        for module in self._modules.values():
            act = module(x)

        return act


class EncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_num: int,
        hidden_size: int,
    ) -> None:
        """
        Encoder layer constructor

        Args:
            model_dim (int): Model dimensions
            head_num (int): Head number
            hidden_size (int): Hidden layer dimension
        """
        super().__init__()

        self.mha = MultiHeadAttention(model_dim, head_num)
        self.ln1 = LayerNorm(model_dim)
        self.ff = PositionalFeedForward(model_dim, hidden_size)
        self.ln2 = LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder layer forward function

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output activations
        """
        x_skip = x
        act = self.mha(q=x, k=x, v=x)
        act = self.ln1(act + x_skip)

        act_skip = act
        act = self.ff(act)
        act = self.ln2(act + act_skip)

        return act
