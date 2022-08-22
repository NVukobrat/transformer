import pytest

import torch
from torch import nn

from transformer.components.layer_norm import LayerNorm
from transformer.components.multi_head_attention import MultiHeadAttention
from transformer.components.positional_feed_forward import PositionalFeedForward
from transformer.components.scaled_dot_product_att import ScaledDotProductAttention

dim_variations = [(1, 1, 3, 9), (1, 1, 9, 3), (1, 3, 3, 9), (2, 3, 3, 9)]


@pytest.mark.parametrize("dim", dim_variations)
def test_positional(dim):
    sdpa = ScaledDotProductAttention()

    q = torch.rand(dim)
    k = torch.rand(dim)
    v = torch.rand(dim)

    scores = sdpa(q, k, v)

    assert scores.shape == v.shape


dim_variations = [4, 8, 16]


@pytest.mark.parametrize("batch_size", dim_variations)
@pytest.mark.parametrize("seq_len", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
@pytest.mark.parametrize("head_num", dim_variations)
def test_multi_head(batch_size, seq_len, model_dim, head_num):
    if model_dim < head_num:
        pytest.skip()  # Number of heads can't be grater then model dim

    inputs_shape = (1, batch_size, seq_len, model_dim)
    q = torch.rand(inputs_shape)
    k = torch.rand(inputs_shape)
    v = torch.rand(inputs_shape)

    mha = MultiHeadAttention(model_dim, head_num)

    attn = mha(q, k, v)

    assert attn.shape == (1, batch_size, seq_len, model_dim)


@pytest.mark.parametrize("batch_size", dim_variations)
@pytest.mark.parametrize("seq_len", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
def test_layer_norm(batch_size, seq_len, model_dim):
    inputs_shape = (1, batch_size, seq_len, model_dim)

    x = torch.rand(inputs_shape)

    framework_layer_norm = nn.LayerNorm(model_dim)
    expected_out = framework_layer_norm(x)

    implemented_layer_norm = LayerNorm(model_dim=model_dim)
    generated_out = implemented_layer_norm(x)

    assert torch.allclose(
        expected_out,
        generated_out,
        rtol=1,  # Due to the nn.LayerNorm empty tensor randomness
        # couldn't match the exact attributes.
        atol=1e-05,
        equal_nan=False,
    )


@pytest.mark.parametrize("batch_size", dim_variations)
@pytest.mark.parametrize("seq_len", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
@pytest.mark.parametrize("hidden_size", dim_variations)
def test_positional_feed_forward(batch_size, seq_len, model_dim, hidden_size):
    inputs_shape = (1, batch_size, seq_len, model_dim)

    x = torch.rand(inputs_shape)

    pff = PositionalFeedForward(model_dim, hidden_size)

    out = pff(x)

    assert out.shape == (1, batch_size, seq_len, model_dim)
