import torch
from torch import nn

from transformer.model.encoder import Encoder
from transformer.model.decoder import Decoder


class Transformer(nn.Module):
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
        Transformer constructor

        Args:
            dict_size (int): Size of the dictionary of embeddings
            model_dim (int): Model dimensions
            max_seq_len (int): Maximum sequence length
            head_num (int): Head number
            hidden_size (int): Hidden layer dimension
            num_layers (int): Number of layers
        """
        super().__init__()

        self.encoder = Encoder(
            dict_size,
            model_dim,
            max_seq_len,
            head_num,
            hidden_size,
            num_layers,
        )

        self.decoder = Decoder(
            dict_size,
            model_dim,
            max_seq_len,
            head_num,
            hidden_size,
            num_layers,
        )
        self.linear = nn.Linear(model_dim, dict_size)
        self.softmax = nn.Softmax()

    def forward(self, enc_inp: torch.Tensor, dec_inp: torch.Tensor) -> torch.Tensor:
        """
        Transformer forward function

        Args:
            enc_inp (torch.Tensor): Encoder inputs
            dec_inp (torch.Tensor): Decoder inputs

        Returns:
            torch.Tensor: Transformers results
        """
        enc_act = self.encoder(enc_inp)
        dec_act = self.decoder(dec_inp, enc_act)

        act = self.linear(dec_act)
        res = self.softmax(act)

        return res
