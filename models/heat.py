"""
Below contains all the necessary functions to get HEAT for posts, assuming the
data is passed in as correct format, and that the `sentence_bert` column is created
already.

Note, the `date` column must be pre-processed.
"""

import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path
from utils.io.data_handler import (
    pad_batched_tensor,
    pad_tensor,
    remove_padding_from_tensor,
)
from utils.io.my_pickler import my_pickler

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from global_parameters import device_to_use

device = torch.device(device_to_use if torch.cuda.is_available() else "cpu")


class HeatLayer(nn.Module):
    """
    Custom PyTorch layer which aggregates historical states
    based on timestamp information and information decay.
    """

    def __init__(
        self, input_shape=768, output_shape=768, epislon_prior=0.01, beta_prior=0.001
    ):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        # Hawkes parameters priors
        self.epsilon = torch.nn.Parameter(
            torch.Tensor(epislon_prior), requires_grad=True
        )
        self.beta = torch.nn.Parameter(torch.Tensor(beta_prior), requires_grad=True)

        # Initialize weights
        # nn.init.normal_(self.weights)
        # nn.init.normal_(self.bias)

    def forward(self, h, t):
        """
        Aggregates the historical states (e.g. the previous hidden states
        in the LSTM of the previous layer). Takes as input the representations
        for the entire timeline, or at the very least the representations of
        the previous historical states (h) and the time-stamps (t) for those
        associated posts.

        Returns HEAT states for all input hidden states. Each HEAT state
        summarizes influence of all previous posts.

        Args:
            h ([type]): [historical hidden states]
            t ([type]): [timestamps]

        Returns:
            [type]: [description]
        """

        print("===== `HeatLayer, forward` =====")
        print("------- test")
        heat_representations = apply_heat_on_matrix_representations(
            h,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=self.epsilon,
            beta=self.beta,
            verbose=False,
            exclude_current_post=False,
            temporal_direction="past",
            remove_padding=True,  # Remove any padding, if desired
            put_back_padding=False,  # Put back padding, if desired
            padding_value=-123.0,
            max_seq_length=124,
        )

        print("===== `HeatLayer, forward` =====")
        print("heat_representations.shape", heat_representations.shape)

        return heat_representations


def extract_time_delta_matrix(t):
    """
    Takes an input vector consisting of timestamps (epoch time, days), and
    returns a time-delta matrix which are pair-wise time-deltas where the i'th
    element in the input vector is compared to each other timestamp, j, in the
    vector, and the time-delta between those are measured for all i and j.

    Element (i, j) in the matrix measures the time-delta between the ith
    timestamp relative to the jth timestamp. It is positive where i >= j, and
    negative otherwise. If the optional parameter (zero_future_posts) is set to
    True, then all negative time-deltas (i.e. future j, relative to i) are set
    as zero.
    """
    # https://stackoverflow.com/questions/52780559/outer-sum-etc-in-pytorch
    tau = t.reshape(-1, 1) - t

    return tau


def pairwise_cos_sim(a, b):
    """
    From: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L23

    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def pairwise_dissimilarity(a, b):
    d = 1 - pairwise_cos_sim(a, b)

    return d


def pairwise_self_dissimilarity(x):
    """
    Takes an input Tensor of post representations, and measures the pair-wise
    dissimalirity matrix of all posts relative to each other.
    """

    d = 1 - pairwise_cos_sim(x, x)

    return d

import torch

# def apply_heat_on_matrix_representations(
#     x,
#     t,
#     epsilon=0.01,
#     beta=0.001,
#     verbose=False,
#     exclude_current_post=False,
#     temporal_direction="past",
#     remove_padding=True,
#     put_back_padding=False,
#     padding_value=-123.0,
#     max_seq_length=124,
#     allow_negative_decay=False,
#     sum_decay_with_original=True,
#     background_intensity="default",
#     delta_representations=False
# ):
#     """
#     Code provided by ChatGPT, which supposedley vectorizes the 2 for loops.
#     """
#     batch_size, max_posts, dimensions = x.shape

#     x_entire_batch = x.clone().to(device)
#     t_entire_batch = t.clone().to(device)

#     if remove_padding:
#         x_entire_batch = remove_padding_from_tensor(
#             x_entire_batch, padding_value=padding_value, return_mask_only=False
#         )
#         x_entire_batch = remove_padding_from_tensor(
#             x_entire_batch, padding_value=0.0, return_mask_only=False
#         )
#         t_entire_batch = t_entire_batch.reshape(batch_size, -1)[:, :max_posts]

#     tau_matrix = extract_time_delta_matrix(t_entire_batch)

#     temporal_deltas, historical_x = get_temporal_deltas_and_historical_x(
#         tau_matrix=tau_matrix,
#         i=torch.arange(max_posts).unsqueeze(0).repeat(batch_size, 1).view(-1),
#         x=x_entire_batch.repeat(1, max_posts, 1).view(-1, dimensions),
#         temporal_direction=temporal_direction,
#         exclude_current_post=exclude_current_post,
#         delta_representations=delta_representations
#     )

#     temporal_deltas = temporal_deltas.to(device)
#     historical_x = historical_x.to(device)

#     if historical_x.shape[0] > 0:
#         decay = epsilon * torch.exp(beta * temporal_deltas)
#         decay = decay.reshape(batch_size, max_posts, max_posts, 1)

#         if allow_negative_decay:
#             non_negative_historical_x = historical_x
#         else:
#             non_negative_historical_x = torch.max(
#                 historical_x, torch.tensor([0.0]).to(device)
#             )

#         if background_intensity == "default":
#             if sum_decay_with_original:
#                 hawkes_out = historical_x + non_negative_historical_x * decay
#             else:
#                 hawkes_out = non_negative_historical_x * decay

#             aggregated = torch.sum(hawkes_out, dim=2)

#         elif background_intensity == "mean_lstm":
#             mean_lstm = torch.mean(x_entire_batch, dim=1)
#             aggregated = mean_lstm + torch.sum(non_negative_historical_x * decay, dim=2)

#     else:
#         aggregated = torch.zeros_like(x_entire_batch[:, 0], dtype=torch.float32, device=device)

#     x_heat = aggregated.to(device)

#     if put_back_padding or batch_size > 1:
#         x_heat = pad_tensor(
#             x_heat, max_seq_length=max_seq_length, padding_value=padding_value
#         )

#     return x_heat

def apply_vectorized_heat_on_matrix_representations():
    pass

def apply_heat_on_matrix_representations(
    x,  # Representations (e.g. sentence-bert embedding)
    t,  # Timestamps (epoch time, e.g. in days)
    epsilon=0.01,
    beta=0.001,
    verbose=False,
    exclude_current_post=False,
    temporal_direction="past",
    remove_padding=True,  # Remove any padding, if desired
    put_back_padding=False,  # Put back padding, if desired
    padding_value=-123.0,
    max_seq_length=124,
    allow_negative_decay=False,
    sum_decay_with_original=True,
    background_intensity="default",
    delta_representations=False
):
    """
    Takes as input `x` which is a matrix, where each row corresponds to the
    representation for the ith post in the timeline. `x` contains the whole
    timeline and is of shape (n_posts, dimensions_of_post_representation).

    The input should not be padded. Padding should be removed before running
    this function.
    """

    # print("=== `apply_heat_on_matrix_representations` ===")

    batch_size = x.shape[0]

    # TODO: may need to move the values to GPU.
    x_entire_batch = x.clone()
    t_entire_batch = t.clone()

    entire_batch_x_heat_list = []
    
    
    # print("Saving... debuging")
    # my_pickler("o", "x_entire_batch", x_entire_batch, folder="debugging")
    # my_pickler("o", "t_entire_batch", t_entire_batch, folder="debugging")
    # print("Saved pickles!!")

    # TODO: vectorize over batches (timelines)
    # Loop over each timeline in the batch (e.g. each user)
    # x = (batch_size, n_posts, 128)
    for batch in range(batch_size):
        # Extract values for current batch
        x = x_entire_batch[batch, :].clone()
        t = t_entire_batch[batch, :].clone()

        if remove_padding:
            x = remove_padding_from_tensor(
                x, padding_value=padding_value, return_mask_only=False
            )
            # TODO: This is a bit of a hack, but it works for now. May cause issues in future.
            x = remove_padding_from_tensor(
                x, padding_value=0.00000, return_mask_only=False
            )

            # Remove padding from time features
            t = t.reshape(
                -1
            )  # Remove the unncessary 1 dimension for time (n_posts, 1) --> (n_posts)
            n_posts = x.shape[0]
            t = t[:n_posts]  # Apply padding mask

        heat_list = (
            []
        )  # Initialize list to store, when iterating over the i'th representation

        tau_matrix = extract_time_delta_matrix(t)  # Pair-wise time-deltas
        
        # my_pickler("o", "tau_matrix", tau_matrix, folder="debugging")
        # print("Saved tau_matrix!!")
        
        # x = (b, n_posts, 128) --> (128)

        # TODO: vectorize over historical posts
        # Loop over each row in x (e.g. each post)
        for i in range(n_posts):

            # Returns time deltas and posts either in the past or future
            temporal_deltas, historical_x = get_temporal_deltas_and_historical_x(
                tau_matrix=tau_matrix,
                i=i,
                x=x,
                temporal_direction=temporal_direction,
                exclude_current_post=exclude_current_post,
                delta_representations=delta_representations
            )

            # Note, this may be costly (only on GPU)
            temporal_deltas = temporal_deltas.to(device)
            historical_x = historical_x.to(device)

            # See if there are actually historical posts
            if historical_x.shape[0] > 0:

                # Time-decay to apply to historical features (vector)
                decay = epsilon * torch.exp(beta * temporal_deltas)
                decay = decay.to(device)

                if allow_negative_decay:
                    non_negative_historical_x = historical_x
                else:  # Returns only non-negative values: e_{k}^{i'} in equation 1.
                    non_negative_historical_x = torch.max(
                        historical_x, torch.tensor([0.0]).to(device)
                    )

                # For broadcasting (multiplying)
                decay = decay[:, None]
                
                if background_intensity == "default":
                    # Hawkes output, for each historical representation
                    if sum_decay_with_original:
                        hawkes_out = historical_x + non_negative_historical_x * decay
                    else:
                        hawkes_out = non_negative_historical_x * decay
                    

                    # Sum historical Hawkes outputs for previous posts, to get
                    # single HEAT representation at each post index (768).
                    aggregated = torch.sum(hawkes_out, axis=0)
                    
                # Treat the background intensity as the mean over the backward/ forward LSTM hidden states.
                elif background_intensity == "mean_lstm":
                    mean_lstm = torch.mean(x, axis=0)
                    aggregated = mean_lstm + torch.sum(non_negative_historical_x * decay, axis=0)

            # If no historical posts, then return aggregated vector of 0s
            else:
                # Return a vector of 0s, the same shape as the first post representation
                aggregated = torch.zeros_like(x[0, :], dtype=float).to(device)

            # Store the heat representation for each index
            heat_list.append(aggregated)

        # Stacked heat representations, for whole timeline  (n_posts, 768)
        if len(heat_list) == 0:
            print("ERROR: No heat representations were created.")
            print("aggregated:\t", aggregated)
            print("heat_list:\t", heat_list)
            print("historical_x.shape:\t", historical_x.shape)
            print("n_posts:\t", n_posts)
            print("x.shape:\t", x.shape)
            print("batch:\t", batch)
            print("x:\t", x)

        # Check the shape of heat_list

        # else:
        x_heat = torch.stack(heat_list)
        x_heat = x_heat.to(device)  # Ensure that it is on the GPU

        if put_back_padding or batch_size > 1:
            x_heat = pad_tensor(
                x_heat, max_seq_length=max_seq_length, padding_value=padding_value
            )
        entire_batch_x_heat_list.append(x_heat)
    entire_batch_x_heat = torch.stack(entire_batch_x_heat_list)

    x_heat = entire_batch_x_heat.to(device)
    
    # my_pickler("o", "x_heat", x_heat, folder="debugging")
    # print("Saved x_heat!!")
    
    # my_pickler("o", "x_heat", x_heat, folder="break this code")


    return x_heat


def apply_differencing_heat_on_matrix_representations(
    x,  # Representations (e.g. sentence-bert embedding)
    t,  # Timestamps (epoch time, e.g. in days)
    epsilon=0.01,
    beta=0.001,
    verbose=False,
    temporal_direction="past",
    remove_padding=True,  # Remove any padding, if desired
    put_back_padding=False,  # Put back padding, if desired
    padding_value=-123.0,
    max_seq_length=124,
    exclude_current_post=False,
    normalize_by_n_posts=True,
):
    """
    Takes as input `x` which is a matrix, where each row corresponds to the
    representation for the ith post in the timeline. `x` contains the whole
    timeline and is of shape (n_posts, dimensions_of_post_representation).

    The input should not be padded. Padding should be removed before running
    this function.
    """

    batch_size = x.shape[0]

    # Loop over each batch (i.e. timeline). Generally should be just 1.
    for batch in range(batch_size):

        # Extract values for current batch
        x = x[batch, :]
        t = t[batch, :]

        # Remove padding for sequence in current batch
        if remove_padding:
            x = remove_padding_from_tensor(
                x, padding_value=padding_value, return_mask_only=False
            )
            t = remove_padding_from_tensor(
                t, padding_value=padding_value, return_mask_only=False
            )

        # Remove the unncessary 1 dimension for time (n_posts, 1) --> (n_posts)
        t = t.reshape(-1)

        # Initialize list to store, when iterating over the i'th representation
        heat_list = []
        # Pair-wise time-deltas
        tau_matrix = extract_time_delta_matrix(t)
        dissimilarity_matrix = pairwise_self_dissimilarity(x)

        # Loop over each row in x (e.g. each post)
        n_posts = x.shape[0]
        for i in range(n_posts):
            # Returns time deltas (vector) and posts either in the past or future
            temporal_deltas, dissimilarity = get_temporal_deltas_and_historical_x(
                tau_matrix=tau_matrix,
                i=i,
                x=dissimilarity_matrix,
                temporal_direction=temporal_direction,
                exclude_current_post=exclude_current_post,
            )

            # Get along just the current
            dissimilarity = dissimilarity[:, 0]

            # See if there are actually historical posts
            if temporal_deltas.shape[0] > 0:

                # Time-decay to apply to historical features (vector)
                decay = epsilon * torch.exp(-beta * temporal_deltas)

                # Hawkes output, for each historical representation
                hawkes_out = dissimilarity * decay

                # Sum historical Hawkes outputs for previous posts, to get
                # single HEAT representation at each post index (768).
                aggregated = torch.sum(hawkes_out, axis=0)
                # Normalize, if desired (normalize, due to sum)
                if normalize_by_n_posts:
                    aggregated /= n_posts

            # If no historical posts, then return aggregated vector of 0s
            else:
                aggregated = torch.tensor(0.0)

            # Store the heat representation for each index
            heat_list.append(aggregated)

        # Stacked heat representations, for whole timeline  (n_posts, 768)
        x_heat = torch.stack(heat_list)

        if put_back_padding:
            x_heat = pad_tensor(
                x_heat, max_seq_length=max_seq_length, padding_value=padding_value
            )

    return x_heat


def get_temporal_deltas_and_historical_x(
    tau_matrix, i, x, temporal_direction="past", exclude_current_post=False, delta_representations=False
):
    """
    Used for getting either future, or past HEAT representation.

    Takes as input the time-delta matrix (n_posts, n_posts), which measures
    pairwise time-deltas between post representations x (n_posts, 768).

    Returns time-deltas and historical posts, based on whether we desire the
    future or past posts.
    """

    # print("==== `get_temporal_deltas_and_historical_x` ====")

    # Time-deltas, relative to the i'th post
    time_deltas = tau_matrix[i, :]

    # Whether to exclude the current post in the "historical posts"
    if temporal_direction == "future":  # Future temporal direction
        if exclude_current_post:
            temporal_deltas = time_deltas[time_deltas < 0]
        else:
            temporal_deltas = time_deltas[time_deltas <= 0]

        temporal_deltas = torch.abs(temporal_deltas)  # Convert negative to positive

        n_future_x = temporal_deltas.shape[0]  # Number of historical posts

        if n_future_x > 0:
            historical_x = x[-n_future_x:, :]  # Slice the future posts
        else:
            historical_x = torch.Tensor([])
            
        if delta_representations:
            # [1, 1, 3, 2, 1] -- > [0, 0, 2, -1, -1]
            # historical_x = np.diff(historical_x, prepend=historical_x[0].to(device)).to(device)  # Starts with 0.
            historical_x = torch.diff(historical_x) 
            print("historical_x.shape", historical_x.shape)
            historical_x = torch.cat([torch.zeros(1, 1).to(device), historical_x], dim=-1)  # Starts with 0.
            print("2. historical_x.shape", historical_x.shape)
            print("historical_x[:3]", historical_x[:3])
    # Past direction
    else:
        if exclude_current_post:
            temporal_deltas = time_deltas[time_deltas > 0]
        else:
            temporal_deltas = time_deltas[time_deltas >= 0]

        n_historical_x = temporal_deltas.shape[0]  # Number of historical posts

        if n_historical_x > 0:
            historical_x = x[:n_historical_x, :]
            
            if delta_representations:
                # [1, 1, 3, 2, 1] -- > [0, 0, 2, -1, -1]
                print("historical_x.shape", historical_x.shape)
                print("historical_x[0].shape", historical_x[0].shape)

                historical_x = torch.diff(historical_x)  
                print("1... historical_x.shape", historical_x.shape)
                historical_x = torch.cat([torch.zeros(1, 1).to(device), historical_x], dim=-1)  # Starts with 0.
                print("2.... historical_x.shape", historical_x.shape)
                # historical_x = np.diff(historical_x, prepend=historical_x[0].to(device)).to(device)  # Starts with 0.
                print("historical_x[:3]", historical_x[:3])
            
        else:
            historical_x = torch.Tensor([])
            
        

    return temporal_deltas, historical_x


# def apply_future_heat_on_matrix_representations():
#     """
#     Applies HEAT, but in reverse direction on future posts rather than
#     historical ones.
#     """


def zero_future_time_deltas(tau, zero_past_instead=False):
    """
    Takes an input matrix, tau, of size (n_posts, n_posts). Elements in tau
    which correspond to future posts (relative to the ith current post) are


    Args:
        tau (_type_): _description_

    Returns:
        _type_: _description_
    """

    return tau


def return_time_deltas_for_current_post(
    timestamps,
    current_t,
    units="days",
    exclude_current_post=True,
    temporal_direction="past",
):
    """
    Returns a Series of the historical posts before the current post to be assessed,
    and the time delta between those.

    Takes as input all timestamps for all posts, and the timestamp for the current
    post to be assessed.

    temporal_direction:
        'past' - only includes historical posts
        'future' - only includes future posts
    """

    # Return historical posts (including, and relative to the post to be assessed)
    if temporal_direction == "past":
        if exclude_current_post:
            historical_posts = timestamps[current_t > timestamps]
        else:
            historical_posts = timestamps[current_t >= timestamps]
    elif temporal_direction == "future":
        if exclude_current_post:
            historical_posts = timestamps[current_t < timestamps]
        else:
            historical_posts = timestamps[current_t <= timestamps]
    # Will return negative values
    elif temporal_direction == "both":
        if exclude_current_post:
            historical_posts = timestamps[current_t != timestamps]
        else:
            historical_posts = timestamps

    # Return the time gap between previous posts to the current post (in days)
    time_deltas = abs(historical_posts - current_t)

    return time_deltas


def return_dimensions_for_concat_heat_representations(
    input_dim=768,
    present_dim=768,
    concat_present=False,
    temporal_direction="past",
    bilstm=False,
    concat_raw=False,
    raw_dim=768
):
    """
    Used for identfying the dimension size of the HEAT representations,
    for a single post.
    """

    output_dim = input_dim

    # Concatenate future and past, if desired
    if temporal_direction == "both":
        output_dim *= 2

    # Concatenate present post, if desired
    if concat_raw:
        if concat_present:
            output_dim += raw_dim
    else:
        if concat_present:
            output_dim += present_dim
            
    return output_dim


def return_dimensions_for_concat_heat_representations_differencing(
    input_dim=768, concat_present=False, temporal_direction="past"
):
    """
    Used for identfying the dimension size of the HEAT representations,
    for a single post.
    """
    dim_of_heat = 1

    # Future, or past
    output_dim = dim_of_heat

    # Concatenate future and past, if desired
    if temporal_direction == "both":
        output_dim += dim_of_heat

    # Concatenate present post, if desired
    if concat_present:
        output_dim += input_dim

    return output_dim


def heat_concatenated_past_present_future(
    x,
    t,
    epsilon,
    beta,
    present_features=None,
    temporal_direction="both",
    concat_present=True,
    exclude_current_post=False,
    remove_padding=True,  # Remove any padding, if desired
    put_back_padding=False,
    padding_value=-123.0,
    max_seq_length=124,
    bidirectional=False,
    temporal_direction_past_only=False,  # If true, then only use past direction on the bilstm states
    allow_negative_decay=False,
    sum_decay_with_original=True,
    background_intensity="default",
    delta_representations=False
):
    """
    TODO: Note, may need to change temporal_direction to past, for the future
    HEAT. Unsure which direction the bilstm future states are outputted in.

    Returns a Tensor, where it concatenates heat representations from past,
    present, and or future.

    Applies the function `apply_heat_on_matrix_representations`, but does this
    optionally twice for past and future. Then concatenate present if desired.

    [past, present, future] is the ordering of the concatenation.

    `exclude_current_post` concerns whether the current post is excluded in the
    heat aggregation, and is seperate from whether you `concatenate_present`.
    """
    # print("==== `heat_concatenated_past_present_future` ====")
    if temporal_direction == "past":
        if bidirectional:
            x_bilstm = x.clone()
            x = x_bilstm[:, :, : x_bilstm.shape[-1] // 2]

        h_past = apply_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction=temporal_direction,
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            allow_negative_decay=allow_negative_decay,
            sum_decay_with_original=sum_decay_with_original,
        )
        
        if concat_present:
            # Add padding back to the number of posts, if needed
            if present_features.shape[1] != h_past.shape[1]:
                present_features = pad_batched_tensor(
                    present_features,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
                h_past = pad_batched_tensor(
                    h_past,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
                concat_h = torch.cat([h_past, present_features], dim=-1)
            else:
                concat_h = torch.cat([h_past, present_features], dim=-1)
        else:
            concat_h = h_past

        concat_h = concat_h.to(device)

    elif temporal_direction == "future":
        if bidirectional:
            x_bilstm = x.clone()
            x = x_bilstm[:, :, x_bilstm.shape[-1] // 2 :]

        h_future = apply_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction=temporal_direction,
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            allow_negative_decay=allow_negative_decay,
            sum_decay_with_original=sum_decay_with_original,
        )
        if concat_present:
            # Add padding back to the number of posts, if needed
            if present_features.shape[1] != h_future.shape[1]:
                present_features = pad_batched_tensor(
                    present_features,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
                h_future = pad_batched_tensor(
                    h_future,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
                concat_h = torch.cat([h_future, present_features], dim=1)
        else:
            concat_h = h

    # Future, and past
    else:
        if bidirectional:
            x_bilstm = x.clone()
            x_past = x_bilstm[:, :, : x_bilstm.shape[-1] // 2]
            x_future = x_bilstm[:, :, x_bilstm.shape[-1] // 2 :]
        else:
            x_past = x.clone()
            x_future = x.clone()

        # Note that x and t are still a batch.
        h_past = apply_heat_on_matrix_representations(
            x_past,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction="past",
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            allow_negative_decay=allow_negative_decay,
            sum_decay_with_original=sum_decay_with_original,
            background_intensity=background_intensity,
            delta_representations=delta_representations
        )

        if concat_present:
            # Add padding back to the number of posts, if needed
            # TODO: would possibly be better to avoid this. Might not reveal bugs.
            if present_features.shape[1] != h_past.shape[1]:
                present_features = pad_batched_tensor(
                    present_features,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
                h_past = pad_batched_tensor(
                    h_past,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
            concat_h = torch.cat(
                [h_past, present_features], dim=-1
            )  # Concatenate along the embedding dimension
        else:
            h_past = pad_batched_tensor(
                h_past,
                max_seq_length=max_seq_length,
                padding_value=padding_value,
            )
            concat_h = h_past

        if temporal_direction_past_only:
            temporal_dir = "past"
        else:
            temporal_dir = "future"

        h_future = apply_heat_on_matrix_representations(
            x_future,  # Representations (e.g. LSTM hidden states)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction=temporal_dir,
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            allow_negative_decay=allow_negative_decay,
            sum_decay_with_original=sum_decay_with_original,
            background_intensity=background_intensity,
            delta_representations=delta_representations
        )

        if concat_present:
            # Add padding back to the number of posts, if needed
            if present_features.shape[1] != h_future.shape[1]:
                h_future = pad_batched_tensor(
                    h_future,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
        else:
            # # Ensure original post representaton, and hidden representations have
            # # same number of posts
            # n_posts = concat_h.shape[1]
            # h_future = h_future[:, :n_posts, :]
            # Add padding back to the number of posts, if needed
            if present_features.shape[1] != h_future.shape[1]:
                h_future = pad_batched_tensor(
                    h_future,
                    max_seq_length=max_seq_length,
                    padding_value=padding_value,
                )
                
        concat_h = concat_h.to(device)
        h_future = h_future.to(device)
        concat_h = torch.cat(
            [concat_h, h_future], dim=-1
        )  # Concatenate along embedding dimension

    # my_pickler(
    #     "o",
    #     "concat_h",
    #     concat_h,
    #     folder="debugging",
    # )

    return concat_h


def differencing_heat_concatenated_past_present_future(
    x,
    t,
    epsilon,
    beta,
    temporal_direction="both",
    concat_present=True,
    exclude_current_post=False,
    remove_padding=True,  # Remove any padding, if desired
    put_back_padding=False,
    padding_value=-123.0,
    max_seq_length=124,
    normalize_by_n_posts=True,
    allow_negative_decay=False,
    sum_decay_with_original=True,
):
    """
    Returns a Tensor, where it concatenates heat representations from past,
    present, and or future.

    Applies the function `apply_heat_on_matrix_representations`, but does this
    optionally twice for past and future. Then concatenate present if desired.

    [past, present, future] is the ordering of the concatenation.

    `exclude_current_post` concerns whether the current post is excluded in the
    heat aggregation, and is seperate from whether you `concatenate_present`.
    """
    if temporal_direction == "past":
        h = apply_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction=temporal_direction,
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            allow_negative_decay=allow_negative_decay,
            sum_decay_with_original=sum_decay_with_original,
        )

        if concat_present:
            if remove_padding:
                x_padding_removed = remove_padding_from_tensor(
                    x, padding_value=padding_value, return_mask_only=False
                )
                concat_h = torch.cat([h, x_padding_removed], dim=1)
            else:
                concat_h = torch.cat([h, x], dim=1)
        else:
            concat_h = h

    elif temporal_direction == "future":
        h = apply_differencing_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction=temporal_direction,
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            normalize_by_n_posts=normalize_by_n_posts,
        )
        if concat_present:
            if remove_padding:
                x_padding_removed = remove_padding_from_tensor(
                    x, padding_value=padding_value, return_mask_only=False
                )
                concat_h = torch.cat([h, x_padding_removed], dim=1)
            else:
                concat_h = torch.cat([h, x], dim=1)
        else:
            concat_h = h

    # Future, and past
    else:
        h_past = apply_differencing_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction="past",
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            normalize_by_n_posts=normalize_by_n_posts,
        )

        if concat_present:
            if remove_padding:
                x_padding_removed = remove_padding_from_tensor(
                    x, padding_value=padding_value, return_mask_only=False
                )
                concat_h = torch.cat([h_past, x_padding_removed], dim=1)
            else:
                concat_h = torch.cat([h_past, x], dim=1)
        else:
            concat_h = h_past

        h_future = apply_differencing_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=epsilon,
            beta=beta,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction="future",
            remove_padding=remove_padding,  # Remove any padding, if desired
            put_back_padding=put_back_padding,  # Put back padding, if desired
            padding_value=padding_value,
            max_seq_length=max_seq_length,
            normalize_by_n_posts=normalize_by_n_posts,
        )
        concat_h = torch.cat([concat_h, h_future], dim=1)

    return concat_h


def heat_on_df(
    user_features,
    epsilon=0.01,
    beta=0.001,
    vectorizer="vectors",
    output_as_dataframe=True,
    verbose=False,
    exclude_current_post=True,
    temporal_direction="past",
):
    """
    Applies heat to all historical posts for a given user.
    Returns a DataFrame for a user, containing the output of
    HEAT for all historical posts, up to that given post.

    Note, beta is sensitive to the units of time that are used. In our case,
    the units are days which are float values.


    Inputs:
    =======
    user_features:
        a dataframe for a given user. Contains the (768) sentence-bert representations of posts,
        and time-deltas for those posts.

    epsilon:
        scalar hyper-parameter which controls amount of excitation.

    beta:
        scalar hyper-parameter which controls rate of decay (forgetting) of information in previous posts.

    temporal_direction:
        'past' - only includes historical posts
        'future' - only includes future posts
    """

    user_features = user_features.copy()

    # User features are pre-processed. Number of posts is number of rows
    post_indexes = user_features.index  # .tolist()

    # Remove missing values

    # Initialize output
    heat_list = []

    # Post indices are chronologically ordered. Loop over them
    for i in post_indexes:

        # Return time-deltas for historical posts, relative to current post looped over
        historical_posts = return_time_deltas_for_current_post(
            user_features["date"],
            user_features["date"][i],
            units="days",
            exclude_current_post=exclude_current_post,
            temporal_direction=temporal_direction,
        )

        # See if there are actually historical posts
        if historical_posts.shape[0] > 0:
            # Join the vector representation for each historical post, with the time-delta
            historical_posts = (
                pd.DataFrame(historical_posts)
                .rename(columns={"date": "time_delta"})
                .join(user_features[[vectorizer, "postid"]])
            )

            # Compute hawkes output, for each historical post
            time = historical_posts["time_delta"]
            feature = historical_posts[vectorizer]  # feature for a historical post

            # Returns a vector (length is number of posts) containing the amount of decay to apply if aggregated
            # at that given post.
            decay = epsilon * np.exp(-beta * time)

            # Return a mask of 1 if element > 0, 0 otherwise.
            # This helps create e_{k}^{i'} in equation 1.
            binary_mask = feature.apply(lambda x: np.greater(x, 0).astype(np.float32))

            # Hawkes output, for each historical post
            hawkes_out = feature + feature * binary_mask * decay

            # Aggregate all Hawkes historical posts, to get HEAT representations
            # Sum historical Hawkes outputs, to get HEAT representation at each
            # post timestamp.
            aggregated = np.sum(hawkes_out, axis=0)

        # If no historical posts, then return aggregated vector of 0s
        else:
            aggregated = np.zeros_like(user_features["vectors"].values[0], dtype=float)

        # Store the heat representation for each post
        heat_list.append(aggregated)
    heat_series = pd.Series(heat_list, index=post_indexes)

    # Return as a Series
    if not output_as_dataframe:
        return heat_series

    # Return as a neatened dataframe
    else:
        user_features["heat"] = heat_series
        if verbose:
            print("User: {}".format(u_id))
        return user_features


def run_heat_on_vectors(
    vectors,
    tl_ids,
    timestamps,
    pids,
    epsilon,
    beta,
    exclude_current_post=True,
    temporal_direction="past",
):

    # ======== HEAT ===========
    print("Applying heat...")

    df = pd.DataFrame(data=[vectors.tolist(), tl_ids, timestamps, pids]).T.rename(
        columns={0: "vectors", 1: "timeline_id", 2: "date", 3: "postid"}
    )
    df["vectors"] = df["vectors"].apply(lambda x: np.array(x))

    # Loop over each timeline (note we only have access to information in their timeline, rather than whole user)
    timeline_ids = df["timeline_id"].unique()
    initialized = True
    for tl_id in timeline_ids:
        timeline_features = df[df["timeline_id"] == tl_id]  # Filter to current timeline

        # Apply heat for a single timeline
        df_single_timeline = heat(
            timeline_features,
            epsilon=epsilon,
            beta=beta,
            vectorizer="vectors",
            output_as_dataframe=True,
            verbose=False,
            exclude_current_post=exclude_current_post,
            temporal_direction=temporal_direction,
        )

        # Concatenate previous operations on previous timelines
        if initialized:
            updated_df = df_single_timeline
            initialized = False
        else:
            updated_df = pd.concat([updated_df, df_single_timeline], axis=0)

    # Update the features to be heat, rather than raw vectors.
    processed_vector = np.stack(updated_df["heat"])
    # ====== End of HEAT ======

    return processed_vector


# class HeatLayer(nn.Module):
#     """
#     Custom PyTorch layer which aggregates historical states
#     based on timestamp information and information decay.
#     """

#     def __init__(self, input_shape=768, output_shape=768, epislon_prior=0.01, beta_prior=0.001):
#         super().__init__()

#         self.input_shape = input_shape
#         self.output_shape = output_shape

#         # Hawkes parameters priors
#         self.epsilon = torch.nn.Parameter(torch.Tensor(epislon_prior), requires_grad=True)
#         self.beta = torch.nn.Parameter(torch.Tensor(beta_prior), requires_grad=True)

#         # Initialize weights
#         # nn.init.normal_(self.weights)
#         # nn.init.normal_(self.bias)

#     def forward(self, h, t):
#         """
#         Aggregates the historical states (e.g. the previous hidden states
#         in the LSTM of the previous layer). Takes as input the representations
#         for the entire timeline, or at the very least the representations of
#         the previous historical states (h) and the time-stamps (t) for those
#         associated posts.

#         Returns HEAT states for all input hidden states. Each HEAT state
#         summarizes influence of all previous posts.

#         Args:
#             h ([type]): [historical hidden states]
#             t ([type]): [timestamps]

#         Returns:
#             [type]: [description]
#         """

#         # Remove padding, if there is any
#         h = remove_padding(h, padding_value=-123.)
#         t = remove_padding(t, padding_value=-123.)

#         # Extract time-deltas, for each post. Matrix of (n_posts, n_posts).
#         # The i'th column corresponds to the time-deltas for all posts
#         # in the timeline, relative to the i'th post.
#         tau = extract_time_delta_matrix(t)

#         # Return hidden states for historical posts, by filtering
#         historical_post_mask = t >= 0  # True for historical posts
#         historical_representations = h[historical_post_mask]

#         # Filter out representations with negative values
#         non_negative_historical_representations = torch.greater(
#             historical_representations, 0
#         )

#         # Measure the amount of decay for each historical post, based on time-delta
#         time_decay = torch.exp(-self.beta * tau)  # Vector, for all historical

#         # Apply information decay to historical representations, and aggregate
#         heat_representations = (
#             historical_representations
#             + self.epsilon * non_negative_historical_representations * time_decay
#         )

#         # Sum for all historical states, so that each node is (h) dimensions. Layer is thus (h, 124)
#         heat_representations = torch.sum(heat_representations, axis=1)

#         # Put back padding, if necessary
#         heat_representations = apply_padding(heat_representations, padding_value=-123, seq_length=124)

#         return heat_representations
