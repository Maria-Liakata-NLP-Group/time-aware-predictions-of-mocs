import sys

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

from utils.io.data_handler import remove_padding_from_tensor


def handle_padding(
    x, to_pack=True, padding_value=-123.0, batch_first=True, dims_with_no_padding=None
):
    """
    Takes an input sequence `x`, and either packs it, if it is already padded
    (so it can be processed more efficiently by PyTorch, by ignoring these
    values). Otherwise, if to_pack=False, then it removes the padded values.
    """
    if to_pack:
        seq_lengths = (x[:, :, 0] != padding_value).sum(axis=1)
        i_packed = pack_padded_sequence(
            x, seq_lengths.to('cpu'), batch_first=True, enforce_sorted=False
        )

        return i_packed  # , dims_with_no_padding  # Mask which shows which are padded

    # Unpacks the sequence, and removes padded values
    if to_pack == False:
        i_unpacked, seq_lengths = pad_packed_sequence(x, batch_first=batch_first)

        return i_unpacked


def remove_padding(x, padding_value=-123.0000):
    seq_lengths = (x[:,:,0] != padding_value).sum(axis=1)
    print("seq_lengths", seq_lengths)
    x = remove_padding_from_tensor(x, seq_lengths)
    
    return x


def seperate_time_from_features(x, time_last_column=True):
    """
    Seperate auxiliary timestamp features from input features
    """
    t = x[:, :, -1]  # Get timestamps
    x = x[:, :, :-1]

    # Add dim to t
    t = torch.unsqueeze(t, -1)

    return x, t
