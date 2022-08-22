import pytest
import torch

from transformer.model.encoder import Encoder
from transformer.model.decoder import Decoder
from transformer.model.transformer import Transformer

dim_variations = [2, 8, 16]


@pytest.mark.parametrize("dict_size", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
@pytest.mark.parametrize("max_seq_len", dim_variations)
@pytest.mark.parametrize("head_num", dim_variations)
@pytest.mark.parametrize("hidden_size", dim_variations)
@pytest.mark.parametrize("num_encoders", dim_variations)
def test_encoder_decoder(
    dict_size, model_dim, max_seq_len, head_num, hidden_size, num_encoders
):
    if model_dim < head_num:
        pytest.skip()  # Number of heads can't be grater then model dim

    encoders = Encoder(
        dict_size, model_dim, max_seq_len, head_num, hidden_size, num_encoders
    )

    input_dict_size = int(torch.randint(1, dict_size, (1,)))
    input_batch_size = int(torch.randint(1, 100, (1,)))
    input_seq_len = int(torch.randint(1, max_seq_len, (1,)))
    x = torch.randint(input_dict_size, (input_batch_size, input_seq_len))

    enc_act = encoders(x)
    assert enc_act.shape == (1, input_batch_size, input_seq_len, model_dim)

    decoders = Decoder(
        dict_size, model_dim, max_seq_len, head_num, hidden_size, num_encoders
    )

    y = torch.randint(input_dict_size, (input_batch_size, input_seq_len))

    dec_act = decoders(y, enc_act)

    assert dec_act.shape == (1, input_batch_size, input_seq_len, model_dim)


@pytest.mark.parametrize("dict_size", dim_variations)
@pytest.mark.parametrize("model_dim", dim_variations)
@pytest.mark.parametrize("max_seq_len", dim_variations)
@pytest.mark.parametrize("head_num", dim_variations)
@pytest.mark.parametrize("hidden_size", dim_variations)
@pytest.mark.parametrize("num_encoders", dim_variations)
def test_transformer(
    dict_size, model_dim, max_seq_len, head_num, hidden_size, num_encoders
):
    if model_dim < head_num:
        pytest.skip()  # Number of heads can't be grater then model dim

    transformer = Transformer(
        dict_size, model_dim, max_seq_len, head_num, hidden_size, num_encoders
    )

    input_dict_size = int(torch.randint(1, dict_size, (1,)))
    input_batch_size = int(torch.randint(1, 100, (1,)))
    input_seq_len = int(torch.randint(1, max_seq_len, (1,)))
    x = torch.randint(input_dict_size, (input_batch_size, input_seq_len))
    y = torch.randint(input_dict_size, (input_batch_size, input_seq_len))

    res = transformer(x, y)

    assert res.shape == (1, input_batch_size, input_seq_len, dict_size)
