import torch
from torch import nn

from transformer.components.multi_head_attention import MultiHeadAttention
from transformer.components.layer_norm import LayerNorm
from transformer.components.positional_feed_forward import PositionalFeedForward
from transformer.embedding.transformer import TransformerEmbedding


class Decoder(nn.Module):
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
        Decoder constructor

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
            DecoderLayer(model_dim, head_num, hidden_size) for i in range(num_layers)
        ]
        self.decoder = SequentialWrapper(*layers)

    def forward(
        self,
        x_dec: torch.Tensor,
        act_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decoder forward function

        Args:
            x_dec (torch.Tensor): Input tensor
            act_enc (torch.Tensor): Encoder input activations

        Returns:
            torch.Tensor: Decoder activations
        """
        act = self.embedding(x=x_dec)
        act = self.decoder(x_dec=act, act_enc=act_enc)

        return act


class SequentialWrapper(nn.Sequential):
    """
    Wrapper for nn.Sequential.

    Overrides default forward function in order to
    support custom arguments.
    """

    def forward(self, x_dec: torch.Tensor, act_enc: torch.Tensor):
        """
        Sequential wrapper forward override function

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output activations

        Args:
            x_dec (torch.Tensor): Decoder input
            act_enc (torch.Tensor): Encoder activations

        Returns:
            _type_: _description_
        """
        act = x_dec
        for module in self._modules.values():
            act = module(act, act_enc)

        return act


class DecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_num: int,
        hidden_size: int,
    ) -> None:
        """
        Decoder layer constructor

        Args:
            model_dim (int): Model dimensions
            head_num (int): Head number
            hidden_size (int): Hidden layer dimension
        """
        super().__init__()

        self.mha1 = MultiHeadAttention(model_dim, head_num)
        self.ln1 = LayerNorm(model_dim)
        self.mha2 = MultiHeadAttention(model_dim, head_num)
        self.ln2 = LayerNorm(model_dim)
        self.ff = PositionalFeedForward(model_dim, hidden_size)
        self.ln3 = LayerNorm(model_dim)

    def forward(
        self,
        x_dec: torch.Tensor,
        act_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decoder forward function

        Args:
            x_dec (torch.Tensor): Input tensor
            act_enc (torch.Tensor): Encoder activations

        Returns:
            torch.Tensor: Output activations
        """
        x_dec_skip = x_dec
        act = self.mha1(q=x_dec, k=x_dec, v=x_dec)
        act = self.ln1(act + x_dec_skip)

        act_skip = act
        act = self.mha2(q=act, k=act_enc, v=act_enc)
        act = self.ln2(act + act_skip)

        act_skip = act
        act = self.ff(act)
        act = self.ln3(act + act_skip)

        return act
