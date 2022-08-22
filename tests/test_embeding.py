import pytest
import torch

from transformer.embedding.token import TokenEmbedding
from transformer.embedding.positional import PositionalEncoding, get_positional_encoding
from transformer.embedding.transformer import TransformerEmbedding

dim_variations = [4, 8, 16]


@pytest.mark.parametrize("seq_len", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
def test_positional(seq_len, model_dim):
    loop_math = get_positional_encoding(seq_len, model_dim)
    loop_math = torch.from_numpy(loop_math).type(torch.float)

    module_vector_math = PositionalEncoding(seq_len, model_dim)
    module_vector_math = module_vector_math.encoding

    assert torch.allclose(
        loop_math, module_vector_math, rtol=1e-05, atol=1e-08, equal_nan=False
    )


@pytest.mark.parametrize("dict_size", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
def test_token(dict_size, model_dim):
    token_emb = TokenEmbedding(dict_size, model_dim)

    assert token_emb.weight.shape == (dict_size, model_dim)


@pytest.mark.parametrize("dict_size", dim_variations)
@pytest.mark.parametrize("max_seq_len", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
def test_transformer(dict_size, max_seq_len, model_dim):
    transformer_embeddings = TransformerEmbedding(dict_size, model_dim, max_seq_len)

    input_dict_size = int(torch.randint(1, dict_size, (1,)))
    input_batch_size = int(torch.randint(1, 100, (1,)))
    input_seq_len = int(torch.randint(1, max_seq_len, (1,)))

    x = torch.randint(input_dict_size, (input_batch_size, input_seq_len))

    trf_emb = transformer_embeddings(x)

    assert trf_emb.shape == (input_batch_size, input_seq_len, model_dim)
