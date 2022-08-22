import torch
from torch import nn

from transformer.components.scaled_dot_product_att import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, head_num: int):
        """
        Multi-head attention module constructor

        Args:
            model_dim (int): Model dimensions
            head_num (int): Head number
        """
        super().__init__()

        self.head_num = head_num
        self.lin_q = nn.Linear(model_dim, model_dim)
        self.lin_k = nn.Linear(model_dim, model_dim)
        self.lin_v = nn.Linear(model_dim, model_dim)
        self.attention = ScaledDotProductAttention()
        self.lin_concat = nn.Linear(model_dim, model_dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-head attention module forward function

        Args:
            q (torch.Tensor): Query vector
            k (torch.Tensor): Key vector
            v (torch.Tensor): Value vector

        Returns:
            torch.Tensor: Multi-head attention scaled dot-product scores
        """
        q = self.lin_q(q)
        k = self.lin_q(k)
        v = self.lin_q(v)

        q = self.__split(q)
        k = self.__split(k)
        v = self.__split(v)

        attn = self.attention(q, k, v)

        conc = self.__concat(attn)
        conc = self.lin_concat(conc)

        return conc.unsqueeze(0)

    def __split(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split tensor to head number

        Args:
            tensor (torch.Tensor): Referent tensor

        Returns:
            torch.Tensor: Splitted tensor
        """
        model_dim = tensor.shape[-1]
        seq_len = tensor.shape[-2]
        batch_size = tensor.shape[-3]

        new_tensor_dim = model_dim // self.head_num

        tensor = tensor.view(batch_size, seq_len, self.head_num, new_tensor_dim)
        tensor = tensor.transpose(1, 2)

        return tensor

    def __concat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Fuse splitted tensors into single score result tensor

        Args:
            tensor (torch.Tensor): Referent tensor

        Returns:
            torch.Tensor: Concatenated score tensor
        """
        tensor_dim = tensor.shape[-1]
        seq_len = tensor.shape[-2]
        head_num = tensor.shape[-3]
        batch_size = tensor.shape[-4]

        model_dim = head_num * tensor_dim

        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(batch_size, seq_len, model_dim)

        return tensor
