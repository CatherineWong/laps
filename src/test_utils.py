"""
test_utils.py
"""
from src.utils import *


def test_untokenized_strings_to_pretrained_embeddings_static():
    strings_tensor = ["i went to the bank", "the river bank is quite nice today"]
    (
        tokens_tensor,
        embedding_tensor,
        attention_mask,
    ) = untokenized_strings_to_pretrained_embeddings(
        strings_tensor,
        embedding_type=EMBEDDING_STATIC,
        embedding_size=EMBEDDING_SIZE_SM,
    )
    assert len(embedding_tensor.size()) == 3
    assert embedding_tensor.size()[0] == len(strings_tensor)

    assert len(attention_mask.size()) == 2
    assert attention_mask.size()[0] == len(strings_tensor)

    assert attention_mask.size() == embedding_tensor.size()[:-1]
    assert (
        sum(attention_mask[-1]).data.item() == 0
    )  # The last one is the longest and should be not
    assert bool(torch.all(embedding_tensor[0][4] == embedding_tensor[1][2]))


def test_untokenized_strings_to_pretrained_embeddings_contextual():
    strings_tensor = ["i went to the bank", "the river bank is quite nice today"]
    (
        tokens_tensor,
        embedding_tensor,
        attention_mask,
    ) = untokenized_strings_to_pretrained_embeddings(
        strings_tensor,
        embedding_type=EMBEDDING_CONTEXTUAL,
        embedding_size=EMBEDDING_SIZE_SM,
    )
    assert len(embedding_tensor.size()) == 3
    assert embedding_tensor.size()[0] == len(strings_tensor)

    assert len(attention_mask.size()) == 2
    assert attention_mask.size()[0] == len(strings_tensor)

    assert attention_mask.size() == embedding_tensor.size()[:-1]
    assert sum(attention_mask[-1]).data.item() == 0
    # Difference in bank
    assert not bool(torch.all(embedding_tensor[0][5] == embedding_tensor[1][3]))
