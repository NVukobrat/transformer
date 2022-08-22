from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Represents kind of a lookup table that stores embeddings (mostly used
    to store word embeddings).
    """

    def __init__(self, dict_size: int, model_dim: int):
        """
        Constructor of token embeddings.

        Args:
            dict_size (int): Size of the dictionary of embeddings
            model_dim (int): The size of each embedding vector
        """
        super().__init__(dict_size, model_dim, padding_idx=1)
