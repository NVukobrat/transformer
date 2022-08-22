from torch import nn

from transformer.embedding.positional import PositionalEncoding
from transformer.embedding.token import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    Adds positional information to the word embeddings.
    """

    def __init__(self, dict_size: int, model_dim: int, max_seq_len: int):
        """
        Transformer embedding constructor.

        Args:
            dict_size (int): Size of the dictionary of embeddings
            model_dim (int): The size of each embedding vector
            max_seq_len (int): Maximum sequence length
        """
        super().__init__()
        self.tok_emb = TokenEmbedding(dict_size, model_dim)
        self.pos_emb = PositionalEncoding(max_seq_len, model_dim)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        trf_emb = tok_emb + pos_emb

        return trf_emb
