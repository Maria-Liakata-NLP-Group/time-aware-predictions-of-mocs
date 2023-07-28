import sys

import torch

sys.path.insert(0, "../../predicting_mocs/")
from utils.time_sensitive_utils import seperate_time_from_features

from models.heat import (
    apply_differencing_heat_on_matrix_representations,
    apply_heat_on_matrix_representations,
    differencing_heat_concatenated_past_present_future,
    heat_concatenated_past_present_future,
    return_dimensions_for_concat_heat_representations,
    return_dimensions_for_concat_heat_representations_differencing,
)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=3):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        print("x.shape", x.shape)
        print("type(x) =", type(x))
        outputs = torch.sigmoid(self.linear(x))

        return outputs


class LogisticRegressionHeat(torch.nn.Module):
    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="past",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Apply heat on timeline features x, to get new timeline features
        x_heat = apply_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=self.epsilon,
            beta=self.beta,
            verbose=False,
            exclude_current_post=self.exclude_current_post,
            temporal_direction=self.temporal_direction,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,  # Put back padding, if desired
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        # Apply linear activation on each row of the timeline features
        outputs = torch.sigmoid(self.linear(x_heat))

        return outputs


class LogisticRegressionHeatConcat(torch.nn.Module):
    """
    Concatenates past  (heat), present (original representation), future (heat)
    representations together, and then makes a prediction.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="past",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = return_dimensions_for_concat_heat_representations(
            input_dim=input_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length
        self.concat_present = concat_present

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Apply heat on timeline features x, to get new timeline features
        x_heat = heat_concatenated_past_present_future(
            x,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        # Apply linear activation on each row of the timeline features
        outputs = torch.sigmoid(self.linear(x_heat))

        return outputs

    def fix_input_dim_and_linear_layer(self):
        self.input_dim = return_dimensions_for_concat_heat_representations(
            input_dim=self.input_dim,
            concat_present=self.concat_present,
            temporal_direction=self.temporal_direction,
        )
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)


class LogisticRegressionDifferencingHeat(torch.nn.Module):
    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="past",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        normalize_by_n_posts=True,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.differencing_dim = 1
        self.linear = torch.nn.Linear(self.differencing_dim, output_dim)

        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length
        self.normalize_by_n_posts = normalize_by_n_posts

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Apply heat on timeline features x, to get new timeline features
        x_heat = apply_differencing_heat_on_matrix_representations(
            x,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=self.epsilon,
            beta=self.beta,
            verbose=False,
            exclude_current_post=self.exclude_current_post,
            temporal_direction=self.temporal_direction,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,  # Put back padding, if desired
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            normalize_by_n_posts=self.normalize_by_n_posts,
        )

        # (n_samples, 1)
        x_heat = x_heat.reshape(-1, 1)

        # Apply linear activation on each row of the timeline features
        outputs = torch.sigmoid(self.linear(x_heat))

        return outputs


class LogisticRegressionDifferencingHeatConcat(torch.nn.Module):
    """
    Concatenates past  (heat), present (original representation), future (heat)
    representations together, and then makes a prediction.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="past",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
        normalize_by_n_posts=True,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = return_dimensions_for_concat_heat_representations_differencing(
            input_dim=input_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length
        self.concat_present = concat_present
        self.normalize_by_n_posts = normalize_by_n_posts

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Apply heat on timeline features x, to get new timeline features
        x_heat = differencing_heat_concatenated_past_present_future(
            x,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            normalize_by_n_posts=self.normalize_by_n_posts,
        )

        # Apply linear activation on each row of the timeline features
        outputs = torch.sigmoid(self.linear(x_heat))

        return outputs

    def fix_input_dim_and_linear_layer(self):
        self.input_dim = return_dimensions_for_concat_heat_representations_differencing(
            input_dim=self.input_dim,
            concat_present=self.concat_present,
            temporal_direction=self.temporal_direction,
        )
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)


# class LogisticRegressionDifferencingHeatNormalizedConcat(torch.nn.Module):
#     """
#     Concatenates past  (heat), present (original representation), future (heat)
#     representations together, and then makes a prediction.
#     """
#     def __init__(self, input_dim=768+1, output_dim=3, epsilon_prior=0.01,
#             beta_prior=0.001, exclude_current_post=False, temporal_direction="past", remove_padding=True, put_back_padding=False, padding_value=-123.,
#             max_seq_length=124, concat_present=False, normalize_by_n_posts=True):
#         super().__init__()
#         input_dim -= 1  # Remove time from input dimension
#         self.input_dim = return_dimensions_for_concat_heat_representations_differencing(input_dim=input_dim,
#                                                    concat_present=concat_present,
#                                                    temporal_direction=temporal_direction)
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#         self.output_dim = output_dim
#         self.epsilon = epsilon_prior
#         self.beta  = beta_prior
#         self.exclude_current_post = exclude_current_post
#         self.temporal_direction = temporal_direction
#         self.remove_padding = remove_padding
#         self.put_back_padding = put_back_padding
#         self.padding_value = padding_value
#         self.max_seq_length = max_seq_length
#         self.concat_present = concat_present
#         self.normalize_by_n_posts = normalize_by_n_posts

#     def forward(self, x):

#         # Seperate auxiliary timestamp features from input features
#         x, t = seperate_time_from_features(x, time_last_column=True)

#         # Apply heat on timeline features x, to get new timeline features
#         x_heat = differencing_heat_concatenated_past_present_future(x,
#                       t,
#                       epsilon=self.epsilon,
#                       beta=self.beta,
#                       temporal_direction=self.temporal_direction,
#                       concat_present=self.concat_present,
#                       exclude_current_post=self.exclude_current_post,
#                       remove_padding=self.remove_padding,  # Remove any padding, if desired
#                       put_back_padding=self.put_back_padding,
#                       padding_value=self.padding_value,
#                       max_seq_length=self.max_seq_length,
#                       normalize_by_n_posts=self.normalize_by_n_posts)

#         # Apply linear activation on each row of the timeline features
#         outputs = torch.sigmoid(self.linear(x_heat))

#         return outputs

#     def fix_input_dim_and_linear_layer(self):
#         self.input_dim = return_dimensions_for_concat_heat_representations_differencing(input_dim=self.input_dim,
#                                                    concat_present=self.concat_present,
#                                                    temporal_direction=self.temporal_direction)
#         self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

# class LogisticRegressionDifferencingHeatUnnormalizedConcat(torch.nn.Module):
#     """
#     Concatenates past  (heat), present (original representation), future (heat)
#     representations together, and then makes a prediction.
#     """
#     def __init__(self, input_dim=768+1, output_dim=3, epsilon_prior=0.01,
#             beta_prior=0.001, exclude_current_post=False, temporal_direction="past", remove_padding=True, put_back_padding=False, padding_value=-123.,
#             max_seq_length=124, concat_present=False, normalize_by_n_posts=False):
#         super().__init__()
#         input_dim -= 1  # Remove time from input dimension
#         self.input_dim = return_dimensions_for_concat_heat_representations_differencing(input_dim=input_dim,
#                                                    concat_present=concat_present,
#                                                    temporal_direction=temporal_direction)
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#         self.output_dim = output_dim
#         self.epsilon = epsilon_prior
#         self.beta  = beta_prior
#         self.exclude_current_post = exclude_current_post
#         self.temporal_direction = temporal_direction
#         self.remove_padding = remove_padding
#         self.put_back_padding = put_back_padding
#         self.padding_value = padding_value
#         self.max_seq_length = max_seq_length
#         self.concat_present = concat_present
#         self.normalize_by_n_posts = normalize_by_n_posts

#     def forward(self, x):

#         # Seperate auxiliary timestamp features from input features
#         x, t = seperate_time_from_features(x, time_last_column=True)

#         # Apply heat on timeline features x, to get new timeline features
#         x_heat = differencing_heat_concatenated_past_present_future(x,
#                       t,
#                       epsilon=self.epsilon,
#                       beta=self.beta,
#                       temporal_direction=self.temporal_direction,
#                       concat_present=self.concat_present,
#                       exclude_current_post=self.exclude_current_post,
#                       remove_padding=self.remove_padding,  # Remove any padding, if desired
#                       put_back_padding=self.put_back_padding,
#                       padding_value=self.padding_value,
#                       max_seq_length=self.max_seq_length,
#                       normalize_by_n_posts=self.normalize_by_n_posts)

#         # Apply linear activation on each row of the timeline features
#         outputs = torch.sigmoid(self.linear(x_heat))

#         return outputs

#     def fix_input_dim_and_linear_layer(self):
#         self.input_dim = return_dimensions_for_concat_heat_representations_differencing(input_dim=self.input_dim,
#                                                    concat_present=self.concat_present,
#                                                    temporal_direction=self.temporal_direction)
#         self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
