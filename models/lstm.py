import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "../../predicting_mocs/")
from models.heat import (
    apply_heat_on_matrix_representations,
    heat_concatenated_past_present_future,
    return_dimensions_for_concat_heat_representations,
)
from utils.time_sensitive_utils import (
    handle_padding,
    remove_padding,
    seperate_time_from_features,
)

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path
from utils.io.data_handler import pad_tensor, remove_padding_from_tensor
from utils.io.my_pickler import my_pickler


class LSTMVanilla(torch.nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=768, output_dim=3, dropout=0):
        # Hidden dim used to be 256. Should search over [64,128,256]
        # [.25, .5, .75]
        super().__init__()

        # Outputs hidden states with dimensionality hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.hyper_params = None  # Default

    def forward(self, x):
        """
        Performs a forward pass in the network.

        Args:
            x (vector): An input matrix consisting of the input sequence (timeline)
            where each element is an embedding vector of each post for a given user.
        """

        # Pre-processing - handle_padding
        padded_x = handle_padding(x, to_pack=True)  # Pack an already padded sequence

        # First layer of LSTM
        lstm1_out, _ = self.lstm1(padded_x)

        # # Second layer of LSTM
        lstm1_out, _ = self.lstm2(lstm1_out)

        lstm_out_unpacked = handle_padding(lstm1_out, to_pack=False)
        # lstm_out_unpacked = self.dropout_layer(lstm_out_unpacked)

        # Fully connected layer
        fc1_out = self.fc1(lstm_out_unpacked)

        # P{redicted probabilities for each possible label
        y_pred = F.log_softmax(fc1_out, dim=1)

        return y_pred

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )
            self.lstm2 = nn.LSTM(
                input_size=h["lstm_hidden_dim_global"],
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']  # No need for dropout on final layer
                # bidirectional=True,
            )
            self.fc1 = nn.Linear(h["lstm_hidden_dim_global"], self.output_dim)
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )

            self.lstm2 = nn.LSTM(
                input_size=h["lstm1_hidden_dim"],
                hidden_size=h["lstm2_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']  # No need for dropout on final layer
                # bidirectional=True,
            )

            self.fc1 = nn.Linear(h["lstm2_hidden_dim"], self.output_dim)

        # if hyper_params['number_of_lstm_layers'] == 2:


class LSTMVanillaSingleLayer(torch.nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=768, output_dim=3, dropout=0):
        # Hidden dim used to be 256. Should search over [64,128,256]
        # [.25, .5, .75]
        super().__init__()

        # Outputs hidden states with dimensionality hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        # self.lstm2 = nn.LSTM(
        #     input_size=self.hidden_dim,
        #     hidden_size=self.hidden_dim,
        #     batch_first=True,
        #     # dropout=0.5,
        #     # bidirectional=True,
        # )
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.hyper_params = None  # Default

    def forward(self, x):
        """
        Performs a forward pass in the network.

        Args:
            x (vector): An input matrix consisting of the input sequence (timeline)
            where each element is an embedding vector of each post for a given user.
        """

        # Pre-processing - handle_padding
        padded_x = handle_padding(x, to_pack=True)  # Pack an already padded sequence

        # First layer of LSTM
        lstm1_out, _ = self.lstm1(padded_x)

        # # # Second layer of LSTM
        # lstm1_out, _ = self.lstm2(lstm1_out)

        lstm_out_unpacked = handle_padding(lstm1_out, to_pack=False)
        # lstm_out_unpacked = self.dropout_layer(lstm_out_unpacked)

        # Fully connected layer
        fc1_out = self.fc1(lstm_out_unpacked)

        # P{redicted probabilities for each possible label
        y_pred = F.log_softmax(fc1_out, dim=1)

        return y_pred

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )
            # self.lstm2 = nn.LSTM(
            #     input_size=h['lstm_hidden_dim_global'],
            #     hidden_size=h['lstm_hidden_dim_global'],
            #     batch_first=True,
            #     # dropout=h['dropout']  # No need for dropout on final layer
            #     # bidirectional=True,
            # )
            self.fc1 = nn.Linear(h["lstm_hidden_dim_global"], self.output_dim)
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # self.lstm2 = nn.LSTM(
            #     input_size=h['lstm1_hidden_dim'],
            #     hidden_size=h['lstm2_hidden_dim'],
            #     batch_first=True,
            #     # dropout=h['dropout']  # No need for dropout on final layer
            #     # bidirectional=True,
            # )

            self.fc1 = nn.Linear(h["lstm2_hidden_dim"], self.output_dim)


class LSTMVanillaConcat(torch.nn.Module):
    """
    Concatenates the present state (original Sentence-BERT embedding).
    """

    def __init__(self, hidden_dim=256, embedding_dim=768, output_dim=3):
        super().__init__()

        # Outputs hidden states with dimensionality hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )

        # Add embedding dim, since concatenate present
        self.fc1 = nn.Linear(self.hidden_dim + self.embedding_dim, self.output_dim)

    def forward(self, x):
        """
        Performs a forward pass in the network.

        Args:
            x (vector): An input matrix consisting of the input sequence (timeline)
            where each element is an embedding vector of each post for a given user.
        """

        # Pre-processing - handle_padding
        padded_x = handle_padding(x, to_pack=True)  # Pack an already padded sequence

        # First layer of LSTM
        lstm1_out, _ = self.lstm1(padded_x)

        # # Second layer of LSTM
        lstm1_out, _ = self.lstm2(lstm1_out)

        # Unpack the padded sequence and remove padded values
        lstm1_out_unpacked = handle_padding(lstm1_out, to_pack=False)

        # Concatenate present, original post to to hidden representations
        batch = 0
        x_padding_removed = remove_padding_from_tensor(
            x, padding_value=-123.0, return_mask_only=False
        )  # [batch, :]
        h = torch.cat(
            [lstm1_out_unpacked, x_padding_removed], dim=2
        )  # Concatenate last dimension

        # Fully connected layer
        fc1_out = self.fc1(h)

        # Predicted probabilities for each possible label
        y_pred = F.log_softmax(fc1_out, dim=1)

        return y_pred

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )
            self.lstm2 = nn.LSTM(
                input_size=h["lstm_hidden_dim_global"],
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']  # No need for dropout on final layer
                # bidirectional=True,
            )
            self.fc1 = nn.Linear(
                h["lstm_hidden_dim_global"] + self.embedding_dim, self.output_dim
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )

            self.lstm2 = nn.LSTM(
                input_size=h["lstm1_hidden_dim"],
                hidden_size=h["lstm2_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )
            self.fc1 = nn.Linear(
                h["lstm2_hidden_dim"] + self.embedding_dim, self.output_dim
            )


class BiLSTM(torch.nn.Module):
    """
    `bilstm` from `model_selector.py`
    """

    def __init__(self, hidden_dim=256, embedding_dim=768, output_dim=3):
        super().__init__()

        # Outputs hidden states with dimensionality hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(
            self.hidden_dim * 2, self.output_dim
        )  # * 2, because bidirectional

    def forward(self, x):
        """
        Performs a forward pass in the network.

        Args:
            x (vector): An input matrix consisting of the input sequence (timeline)
            where each element is an embedding vector of each post for a given user.
        """

        # Pre-processing - handle_padding
        padded_x = handle_padding(x, to_pack=True)  # Pack an already padded sequence

        # First layer of LSTM
        lstm1_out, _ = self.lstm1(padded_x)

        # Second layer of LSTM
        lstm1_out, _ = self.lstm2(lstm1_out)

        # Unpack the padded sequence and remove padded values
        lstm1_out_unpacked = handle_padding(lstm1_out, to_pack=False)

        # Fully connected layer
        fc1_out = self.fc1(lstm1_out_unpacked)

        # Predicted probabilities for each possible label
        y_pred = F.log_softmax(fc1_out, dim=1)

        return y_pred

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=True,
            )
            self.lstm2 = nn.LSTM(
                input_size=h["lstm_hidden_dim_global"] * 2,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']  # No need for dropout on final layer
                bidirectional=True,
            )
            self.fc1 = nn.Linear(h["lstm_hidden_dim_global"] * 2, self.output_dim)
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=True,
            )

            self.lstm2 = nn.LSTM(
                input_size=h["lstm1_hidden_dim"] * 2,  # *2, because bi-directional
                hidden_size=h["lstm2_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout'],
                bidirectional=True,
            )
            self.fc1 = nn.Linear(h["lstm2_hidden_dim"] * 2, self.output_dim)


class BiLSTMSingleLayer(torch.nn.Module):
    def __init__(self, hidden_dim=256, embedding_dim=768, output_dim=3):
        super().__init__()

        # Outputs hidden states with dimensionality hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(
            self.hidden_dim * 2, self.output_dim
        )  # * 2, because bidirectional

    def forward(self, x):
        """
        Performs a forward pass in the network.

        Args:
            x (vector): An input matrix consisting of the input sequence (timeline)
            where each element is an embedding vector of each post for a given user.
        """

        # Pre-processing - handle_padding
        padded_x = handle_padding(x, to_pack=True)  # Pack an already padded sequence

        # First layer of LSTM
        lstm1_out, _ = self.lstm1(padded_x)

        # # Second layer of LSTM
        # lstm1_out, _ = self.lstm2(lstm1_out)

        # Unpack the padded sequence and remove padded values
        lstm1_out_unpacked = handle_padding(lstm1_out, to_pack=False)

        # Fully connected layer
        fc1_out = self.fc1(lstm1_out_unpacked)

        # P{redicted probabilities for each possible label
        y_pred = F.log_softmax(fc1_out, dim=1)

        return y_pred

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout'],
                bidirectional=True,
            )
            # self.lstm2 = nn.LSTM(
            #     input_size=h['lstm_hidden_dim_global']*2,
            #     hidden_size=h['lstm_hidden_dim_global'],
            #     batch_first=True,
            #     # dropout=h['dropout']  # No need for dropout on final layer
            #     bidirectional=True,
            # )
            self.fc1 = nn.Linear(h["lstm_hidden_dim_global"] * 2, self.output_dim)
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout'],
                bidirectional=True,
            )

            # self.lstm2 = nn.LSTM(
            #     input_size=h['lstm1_hidden_dim']*2,  # *2, because bi-directional
            #     hidden_size=h['lstm2_hidden_dim'],
            #     batch_first=True,
            #     # dropout=h['dropout'],
            #     bidirectional=True
            # )
            self.fc1 = nn.Linear(h["lstm1_hidden_dim"] * 2, self.output_dim)


class BiLSTMConcat(torch.nn.Module):
    def __init__(self, hidden_dim=256, embedding_dim=768, output_dim=3):
        super().__init__()

        # Outputs hidden states with dimensionality hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            bidirectional=True,
        )
        # * 2, because bidirectional, and + embedding dimension, due to concat present
        self.fc1 = nn.Linear(self.hidden_dim * 2 + self.embedding_dim, self.output_dim)

    def forward(self, x):
        """
        Performs a forward pass in the network.

        Args:
            x (vector): An input matrix consisting of the input sequence (timeline)
            where each element is an embedding vector of each post for a given user.
        """

        # Pre-processing - handle_padding
        padded_x = handle_padding(x, to_pack=True)  # Pack an already padded sequence

        # First layer of LSTM
        lstm1_out, _ = self.lstm1(padded_x)

        # Second layer of LSTM
        lstm1_out, _ = self.lstm2(lstm1_out)

        # Unpack the padded sequence and remove padded values
        lstm1_out_unpacked = handle_padding(lstm1_out, to_pack=False)

        # Concatenate present, original post to to hidden representations
        batch = 0
        x_padding_removed = remove_padding_from_tensor(
            x, padding_value=-123.0, return_mask_only=False
        )  # [batch, :]
        h = torch.cat(
            [lstm1_out_unpacked, x_padding_removed], dim=2
        )  # Concatenate last dimension

        # Fully connected layer
        fc1_out = self.fc1(h)

        # P{redicted probabilities for each possible label
        y_pred = F.log_softmax(fc1_out, dim=1)

        return y_pred

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=True,
            )
            self.lstm2 = nn.LSTM(
                input_size=h["lstm_hidden_dim_global"] * 2,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']  # No need for dropout on final layer
                bidirectional=True,
            )
            self.fc1 = nn.Linear(
                h["lstm_hidden_dim_global"] * 2 + self.embedding_dim, self.output_dim
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=True,
            )

            self.lstm2 = nn.LSTM(
                input_size=h["lstm1_hidden_dim"] * 2,  # bi-directional
                hidden_size=h["lstm2_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout'],
                bidirectional=True,
            )

            self.fc1 = nn.Linear(
                h["lstm2_hidden_dim"] * 2 + self.embedding_dim, self.output_dim
            )


class LSTMHeat(torch.nn.Module):
    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="past",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        dropout=0,
    ):
        """
        Define network topography.
        
        Single layer LSTM, with HEAT on past direction.
        """
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        # Outputs hidden states with dimensionality hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=self.dropout,
            # bidirectional=True,
        )
        # self.lstm2 = nn.LSTM(
        #     input_size=self.hidden_dim,
        #     hidden_size=self.hidden_dim,
        #     batch_first=True,
        #     # dropout=0.5,
        #     # bidirectional=True,
        # )
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        # self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):
        """
        Performs a forward pass in the network.

        Args:
            x (vector): An input matrix consisting of the input sequence
            where each element is an embedding vector of each post for a given user.
        """

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # # Second layer of LSTM
        # lstm1_out, _ = self.lstm2(lstm1_out)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # lstm1_out_unpacked = self.dropout_layer(lstm1_out_unpacked)

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = apply_heat_on_matrix_representations(
            lstm1_out_unpacked,  # Representations (e.g. sentence-bert embedding)
            t,  # Timestamps (epoch time, e.g. in days)
            epsilon=self.epsilon,
            beta=self.beta,
            verbose=False,
            exclude_current_post=self.exclude_current_post,
            temporal_direction=self.temporal_direction,
            remove_padding=True,  # Remove any padding, if desired
            put_back_padding=False,  # Put back padding, if desired
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )
        
        # d = {}
        # d['h_heat'] = h_heat
        # d['t'] = t
        # d['x'] = x
        # d['lstm1_out_unpacked'] = lstm1_out_unpacked
        # d['epsilon'] = self.epsilon
        # d['beta'] = self.beta
        
        # my_pickler('o', 'reddit_lstm_heat_past_only_dict_debugging', d, folder='debugging')
        # print('!!!!!!!')
        # exit()

        # Remove padded values from heat

        fc1_out = self.fc1(h_heat)
        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params
        
        # print('h-----\n', h)
            
        self.temporal_direction = "past"
        self.epsilon = h["epsilon_prior"]
        self.beta = h["beta_prior"]
        self.concat_present = False

        # # Fix input dim and linear layer
        # self.heat_output_dim = return_dimensions_for_concat_heat_representations(
        #     input_dim=768,
        #     present_dim=768,  
        #     concat_present=self.concat_present,
        #     temporal_direction=self.temporal_direction,
        # )
        
        self.lstm1 = nn.LSTM(
                input_size=768,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=False,
            )

        # self.fc_input_dim = h["lstm_hidden_dim_global"] # + 768  # + 768 is for concatenation with v

        # self.fc1 = nn.Linear(self.fc_input_dim, self.output_dim)
        

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )
            # self.lstm2 = nn.LSTM(
            #     input_size=h["lstm_hidden_dim_global"],
            #     hidden_size=h["lstm_hidden_dim_global"],
            #     batch_first=True,
            #     # dropout=h['dropout']  # No need for dropout on final layer
            #     # bidirectional=True,
            # )
            self.fc1 = nn.Linear(h["lstm_hidden_dim_global"], self.output_dim)
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )

            # self.lstm2 = nn.LSTM(
            #     input_size=h["lstm1_hidden_dim"],
            #     hidden_size=h["lstm2_hidden_dim"],
            #     batch_first=True,
            #     # dropout=h['dropout']
            #     # bidirectional=True,
            # )

            self.fc1 = nn.Linear(h["lstm1_hidden_dim"], self.output_dim)


class LSTMHeatConcat(torch.nn.Module):
    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):
        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Second layer of LSTM
        lstm1_out, _ = self.lstm2(lstm1_out)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        # print("1111. x.shape: ", x.shape)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    # def fix_input_dim_and_linear_layer(self):
    # self.input_dim = return_dimensions_for_concat_heat_representations(input_dim=self.hidden_dim,
    #                                            concat_present=self.concat_present,
    #                                            temporal_direction=self.temporal_direction)
    # self.fc1 = torch.nn.Linear(self.input_dim, self.output_dim)

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )
            self.lstm2 = nn.LSTM(
                input_size=h["lstm_hidden_dim_global"],
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']  # No need for dropout on final layer
                # bidirectional=True,
            )
            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )

            self.lstm2 = nn.LSTM(
                input_size=h["lstm1_hidden_dim"],
                hidden_size=h["lstm2_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm2_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LSTMHeatConcatSingleLayer(torch.nn.Module):
    """
    `lstm_concat_exclude_present_in_heat_past_present_future_1_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        # Remove padding from HEAT hidden states, from dim 1
        # h_heat = remove_padding_from_tensor(
        #     h_heat,
        #     padding_value=-123.0,
        #     return_mask_only=False,
        # )
        # h_heat = remove_padding_from_tensor(
        #     h_heat, padding_value=0.0, return_mask_only=False
        # )

        # print("(pad removed) h_heat.shape: ", h_heat.shape)

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        # # Export data for debugging
        # export_data = {
        #     "h_heat": h_heat,
        #     "x": x,
        #     "predicted_moc_encoding": predicted_moc_encoding,
        #     "lstm1_out_unpacked": lstm1_out_unpacked,
        #     "t": t,
        #     "beta": self.beta,
        #     "epsilon": self.epsilon,
        #     "fc1_out": fc1_out,
        # }
        # my_pickler("o", "lstm_heat_export_dict", export_data, folder="debugging")

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LSTMHeatWithoutHeatLayer(torch.nn.Module):
    """
    `lstm_concat_exclude_present_in_heat_past_present_future_1_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # # Apply heat on timeline features (unpadded LSTM hidden states)
        # h_heat = heat_concatenated_past_present_future(
        #     lstm1_out_unpacked,
        #     t,
        #     epsilon=self.epsilon,
        #     beta=self.beta,
        #     present_features=x,
        #     temporal_direction=self.temporal_direction,
        #     concat_present=self.concat_present,
        #     exclude_current_post=self.exclude_current_post,
        #     remove_padding=self.remove_padding,  # Remove any padding, if desired
        #     put_back_padding=self.put_back_padding,
        #     padding_value=self.padding_value,
        #     max_seq_length=self.max_seq_length,
        # )

        fc1_out = self.fc1(lstm1_out_unpacked)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(h["lstm_hidden_dim_global"], self.output_dim)


class LSTMHeatConcatPresentLSTMSingleLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        self.heat_output_dim = h_heat.shape[-1]
        print("Heat output dim: ", self.heat_output_dim)
        print("h_heat shape: ", h_heat.shape)
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )
            # dropout=h['dropout']
            # bidirectional=True,
            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class RamitHeat(torch.nn.Module):
    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=input_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=self.heat_output_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

        self.batch_size = 1

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
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

        h_heat = h_heat.unsqueeze(0)  # Add batch dimension

        # Pre-processing - pack padding
        padded_x = handle_padding(h_heat, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        fc1_out = self.fc1(lstm1_out_unpacked)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=self.input_dim,
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

            self.lstm1 = nn.LSTM(
                input_size=self.heat_output_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )
            # dropout=h['dropout']
            # bidirectional=True,

            self.hidden_dim = h["lstm_hidden_dim_global"]
        else:
            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=self.input_dim,
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

            self.lstm1 = nn.LSTM(
                input_size=self.heat_output_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            self.hidden_dim = h["lstm1_hidden_dim"]

        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)


class LSTMHeatConcatLearnableWeights(torch.nn.Module):
    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        # self.epsilon = epsilon_prior
        # self.beta  = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

        # Learnable parameters
        self.epsilon = torch.nn.Parameter(
            data=torch.tensor([epsilon_prior]), requires_grad=True
        )
        self.beta = torch.nn.Parameter(
            data=torch.tensor([beta_prior]), requires_grad=True
        )
        # self.epsilon = torch.nn.Parameter(data=epsilon_prior, requires_grad=True)
        # self.beta = torch.nn.Parameter(data=beta_prior, requires_grad=True)
        # self.epsilon = torch.nn.Parameter(requires_grad=True)
        # self.beta = torch.nn.Parameter(requires_grad=True)

    def forward(self, x):

        # # Parameters must be strictly positive
        # with torch.no_grad():
        #     torch.clamp(self.epsilon, min=0.)
        #     torch.clamp(self.beta, min=0.)

        # self.epsilon = torch.abs(self.epsilon)
        # self.beta = torch.abs(self.beta)

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Second layer of LSTM
        lstm1_out, _ = self.lstm2(lstm1_out)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
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

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        # print('epsilon:', self.epsilon)
        # print('beta:', self.beta)
        # print()

        return predicted_moc_encoding

    # def fix_input_dim_and_linear_layer(self):
    # self.input_dim = return_dimensions_for_concat_heat_representations(input_dim=self.hidden_dim,
    #                                            concat_present=self.concat_present,
    #                                            temporal_direction=self.temporal_direction)
    # self.fc1 = torch.nn.Linear(self.input_dim, self.output_dim)

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"],
            )
            # bidirectional=True,)
            self.lstm2 = nn.LSTM(
                input_size=h["lstm_hidden_dim_global"],
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )
            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"]
                # bidirectional=True,
            )

            self.lstm2 = nn.LSTM(
                input_size=h["lstm1_hidden_dim"],
                hidden_size=h["lstm2_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm2_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatPresentSingleLayer(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatPresentSingleLayerIncludeCurrentPost(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatWithoutHeat(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Debugging:
        h_heat = lstm1_out_unpacked

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatPresentTwoLayers(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm_out, _ = self.lstm1(padded_x)

        # Get LSTM hidden states (still padded)
        lstm_out, _ = self.lstm2(lstm1_out)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm_out_unpacked = handle_padding(
            lstm_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatPresentSingleLayerTemporalDirectionPastOnly(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
            temporal_direction_past_only=True,  # Only use past timestamps for heat, on bilstm states
        )

        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LSTMHeatConcatSingleLayerWithLinearLayerTanh(torch.nn.Module):
    """
    `lstm_concat_exclude_present_in_heat_past_present_future_1_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = 768
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        # Also ideally want some other data, for example the user ids, ground truth labels. However, this is not necessary.
        # Instead need to just see if the representations are calculated correctly.
        # export_data = {
        #     "h_heat": h_heat,
        #     "x": x,
        #     "predicted_moc_encoding": predicted_moc_encoding,
        #     "lstm1_out_unpacked": lstm1_out_unpacked,
        #     "t": t,
        #     "beta": self.beta,
        #     "epsilon": self.epsilon,
        #     "fc1_out": fc1_out,
        # }
        # my_pickler("o", "tanh_activation_function", export_data, folder="debugging")

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatSingleLayerWithLinearLayerTanh(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = 768
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatBiLSTMSingleLayerWithLinearLayerTanh(torch.nn.Module):
    """
    This is the base HEAT that was reported in the paper.
    
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )
        
        # d = {}
        # d['h_heat'] = h_heat
        # d['t'] = t
        # d['x'] = x
        # d['lstm_out'] = lstm1_out_unpacked
        # d['epsilon'] = self.epsilon
        # d['beta'] = self.beta
        
        # my_pickler('o', 'reddit_heat_dict_debugging', d, folder='debugging')
        # print('!!!!!!!')


        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatRawWithLinearLayerTanh(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # print("lstm1_out_unpacked.shape = ", lstm1_out_unpacked.shape)
        # print("x.shape = ", x.shape)

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):

        h = self.hyper_params

        self.exclude_current_post = True
        self.temporal_direction = "both"
        self.epsilon = h["epsilon_prior"]
        self.beta = h["beta_prior"]
        self.dropout = get_default_hyper_params(h, "dropout")
        self.concat_present = True

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,            
                concat_raw=True,
                raw_dim=768
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
                concat_raw=True,
                raw_dim=768
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatConcatBiLSTMTwoLayersWithLinearLayerTanh(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            batch_first=True,
            # dropout=0.5,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Second layer of LSTM
        lstm1_out, _ = self.lstm2(lstm1_out)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=True,
            )
            self.lstm2 = nn.LSTM(
                input_size=h["lstm_hidden_dim_global"] * 2,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                # dropout=h['dropout']  # No need for dropout on final layer
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=True,
            )

            self.lstm2 = nn.LSTM(
                input_size=h["lstm1_hidden_dim"] * 2,  # *2, because bi-directional
                hidden_size=h["lstm2_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout'],
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LearnableHeatLSTMHeatConcatSingleLayerWithLinearLayerTanh(torch.nn.Module):
    """
    `lstm_concat_exclude_present_in_heat_past_present_future_1_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)
        self.epsilon = epsilon_prior
        self.beta = beta_prior

        # # Learnable HEAT parameters
        # self.epsilon = nn.Parameter(
        #     data=torch.Tensor(epsilon_prior), requires_grad=True
        # )
        # self.beta = nn.Parameter(data=torch.Tensor(beta_prior), requires_grad=True)

        # HEAT hyper-parameters parameters
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = 768
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        # Also ideally want some other data, for example the user ids, ground truth labels. However, this is not necessary.
        # Instead need to just see if the representations are calculated correctly.
        # export_data = {
        #     "h_heat": h_heat,
        #     "x": x,
        #     "predicted_moc_encoding": predicted_moc_encoding,
        #     "lstm1_out_unpacked": lstm1_out_unpacked,
        #     "t": t,
        #     "beta": self.beta,
        #     "epsilon": self.epsilon,
        #     "fc1_out": fc1_out,
        # }
        # my_pickler("o", "tanh_activation_function", export_data, folder="debugging")

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        # self.epsilon = h["epsilon_prior"]
        # self.beta = h["beta_prior"]

        # print("==== `apply_hyperparameters` =====")
        # print("self.epsilon: ", self.epsilon)
        # print("self.epsilon.shape:", self.epsilon.shape)
        # print()
        # print("self.beta: ", self.beta)
        # print("self.beta.shape:", self.beta.shape)
        # print("==================================")

        # params = list(net.parameters())
        # params.extend(list(loss.parameters()))

        # Hyper-parameters
        self.exclude_current_post = True
        self.temporal_direction = "both"
        self.exclude_current_post = True
        self.dropout = get_default_hyper_params(h, "dropout")
        self.concat_present = True

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LearnableHeatSoftplusBetaEpsilonLSTMHeatConcatSingleLayerWithLinearLayerTanh(
    torch.nn.Module
):
    """
    `lstm_concat_exclude_present_in_heat_past_present_future_1_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)
        self.epsilon = epsilon_prior
        self.beta = beta_prior

        # # Learnable HEAT parameters
        # self.epsilon = nn.Parameter(
        #     data=torch.Tensor(epsilon_prior), requires_grad=True
        # )
        # self.beta = nn.Parameter(data=torch.Tensor(beta_prior), requires_grad=True)

        # HEAT hyper-parameters parameters
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Enforce HEAT parameters to be positive
        activation_function = nn.Softplus()
        positive_epsilon = activation_function(self.epsilon)
        positive_beta = activation_function(self.beta)

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=positive_epsilon,
            beta=positive_beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = 768
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        # Also ideally want some other data, for example the user ids, ground truth labels. However, this is not necessary.
        # Instead need to just see if the representations are calculated correctly.
        # export_data = {
        #     "h_heat": h_heat,
        #     "x": x,
        #     "predicted_moc_encoding": predicted_moc_encoding,
        #     "lstm1_out_unpacked": lstm1_out_unpacked,
        #     "t": t,
        #     "beta": self.beta,
        #     "epsilon": self.epsilon,
        #     "fc1_out": fc1_out,
        # }
        # my_pickler("o", "tanh_activation_function", export_data, folder="debugging")

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        # Hyper-parameters
        self.exclude_current_post = True
        self.temporal_direction = "both"
        self.exclude_current_post = True
        self.dropout = get_default_hyper_params(h, "dropout")
        self.concat_present = True

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LearnableHeatSigmoidBetaEpsilonLSTMHeatConcatSingleLayerWithLinearLayerTanh(
    torch.nn.Module
):
    """
    `lstm_concat_exclude_present_in_heat_past_present_future_1_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)
        self.epsilon = epsilon_prior
        self.beta = beta_prior

        # # Learnable HEAT parameters
        # self.epsilon = nn.Parameter(
        #     data=torch.Tensor(epsilon_prior), requires_grad=True
        # )
        # self.beta = nn.Parameter(data=torch.Tensor(beta_prior), requires_grad=True)

        # HEAT hyper-parameters parameters
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Enforce HEAT parameters to be positive
        activation_function = nn.Sigmoid()
        positive_epsilon = activation_function(self.epsilon)
        positive_beta = activation_function(self.beta)

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=positive_epsilon,
            beta=positive_beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = 768
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        # Also ideally want some other data, for example the user ids, ground truth labels. However, this is not necessary.
        # Instead need to just see if the representations are calculated correctly.
        # export_data = {
        #     "h_heat": h_heat,
        #     "x": x,
        #     "predicted_moc_encoding": predicted_moc_encoding,
        #     "lstm1_out_unpacked": lstm1_out_unpacked,
        #     "t": t,
        #     "beta": self.beta,
        #     "epsilon": self.epsilon,
        #     "fc1_out": fc1_out,
        # }
        # my_pickler("o", "tanh_activation_function", export_data, folder="debugging")

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        # Hyper-parameters
        self.exclude_current_post = True
        self.temporal_direction = "both"
        self.exclude_current_post = True
        self.dropout = get_default_hyper_params(h, "dropout")
        self.concat_present = True

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatAllowNegativeConcatBiLSTMSingleLayerWithLinearLayerTanh(
    torch.nn.Module
):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
            allow_negative_decay=True,
            sum_decay_with_original=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class BiLSTMHeatAllowNegativeNoSummationConcatBiLSTMSingleLayerWithLinearLayerTanh(
    torch.nn.Module
):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
            allow_negative_decay=True,
            sum_decay_with_original=False,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LearnableHeatSoftplusBetaEpsilonAllowNegativeNoSummationLSTMHeatConcatSingleLayerWithLinearLayerTanh(
    torch.nn.Module
):
    """
    `lstm_concat_exclude_present_in_heat_past_present_future_1_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.5,
            # bidirectional=True,
        )
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)
        self.epsilon = epsilon_prior
        self.beta = beta_prior

        # # Learnable HEAT parameters
        # self.epsilon = nn.Parameter(
        #     data=torch.Tensor(epsilon_prior), requires_grad=True
        # )
        # self.beta = nn.Parameter(data=torch.Tensor(beta_prior), requires_grad=True)

        # HEAT hyper-parameters parameters
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Enforce HEAT parameters to be positive
        activation_function = nn.Softplus()
        positive_epsilon = activation_function(self.epsilon)
        positive_beta = activation_function(self.beta)

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=positive_epsilon,
            beta=positive_beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            allow_negative_decay=True,
            sum_decay_with_original=False,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = 768
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        # Also ideally want some other data, for example the user ids, ground truth labels. However, this is not necessary.
        # Instead need to just see if the representations are calculated correctly.
        # export_data = {
        #     "h_heat": h_heat,
        #     "x": x,
        #     "predicted_moc_encoding": predicted_moc_encoding,
        #     "lstm1_out_unpacked": lstm1_out_unpacked,
        #     "t": t,
        #     "beta": self.beta,
        #     "epsilon": self.epsilon,
        #     "fc1_out": fc1_out,
        # }
        # my_pickler("o", "tanh_activation_function", export_data, folder="debugging")

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        # Hyper-parameters
        self.exclude_current_post = True
        self.temporal_direction = "both"
        self.exclude_current_post = True
        self.dropout = get_default_hyper_params(h, "dropout")
        self.concat_present = True

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                # bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class LearnableHeatAllowNegativeBiLSTMHeatConcatSingleLayerWithLinearLayerTanh(
    torch.nn.Module
):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
            allow_negative_decay=True,
            sum_decay_with_original=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class HeatBiLSTMNoConcatenationOnlyHeatTanh(torch.nn.Module):
    """ """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=None,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
            allow_negative_decay=True,
            sum_decay_with_original=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        # h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        # h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        # h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [
                h_past,
                #  h_present,
                h_future,
            ],
            dim=-1,
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        self.concat_present = False

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class HeatBiLSTMNoConcatenationOnlyHeat(torch.nn.Module):
    """ """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=None,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        # h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        # h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        # scaler = nn.Tanh()
        # h_past = scaler(h_past)
        # # h_present = scaler(h_present)
        # h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [
                h_past,
                #  h_present,
                h_future,
            ],
            dim=-1,
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        self.concat_present = False

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


class HeatNoMSD(torch.nn.Module):
    """
    Remove modelling sequence dynamics. HEAT operates only on previous hidden
    raw posts. No (BiLSTM).
    Tanh scaling of HEAT outputs. Scaled outputs
    are then passed in a single linear layer. Basically logreg.
    
    ----
    Remove bidirectionality, so HEAT operates only on previous hidden states
    of an LSTM.
    
    Concatenation of present LSTM. 
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length
        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)
    
        # Apply heat on timeline features
        h_heat = heat_concatenated_past_present_future(
            x,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=False,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=False,  # Not a BiLSTM input. Is raw posts.
        )
        
        # Apply the scaling
        scaler = nn.Tanh()
        h = scaler(h_heat)
        
        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h)
    
        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params
        
        self.temporal_direction = "both"

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        self.concat_present = False

        # Fix input dim and linear layer
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=768,
            present_dim=768,  
            concat_present=False,
            temporal_direction=self.temporal_direction,
        )
        
        # self.lstm1 = nn.LSTM(
        #         input_size=self.heat_output_dim,
        #         hidden_size=h["lstm_hidden_dim_global"],
        #         batch_first=True,
        #         dropout=h["dropout"],
        #         bidirectional=True,
        #     )

        # self.fc_input_dim = h["lstm_hidden_dim_global"] * 2 + 768  # + 768 is for concatenation with v
        
        self.fc_input_dim = self.heat_output_dim

        self.fc1 = nn.Linear(self.fc_input_dim, self.output_dim)


class HeatNoMSDConcatV(torch.nn.Module):
    """
    Remove modelling sequence dynamics. HEAT operates only on previous hidden
    raw posts. No (BiLSTM).
    Tanh scaling of HEAT outputs. Scaled outputs
    are then passed in a single linear layer. Basically logreg.
    
    ----
    Remove bidirectionality, so HEAT operates only on previous hidden states
    of an LSTM.
    
    Concatenation of present LSTM. 
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length
        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)
    
        # Apply heat on timeline features
        h_heat = heat_concatenated_past_present_future(
            x,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=x,
            temporal_direction=self.temporal_direction,
            concat_present=True,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=False,  # Not a BiLSTM input. Is raw posts.
        )
        
        # Apply the scaling
        scaler = nn.Tanh()
        h = scaler(h_heat)
        
        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h)
    
        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params
        
        self.temporal_direction = "both"

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        self.concat_present = True

        # Fix input dim and linear layer
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=768,
            present_dim=768,  
            concat_present=True,
            temporal_direction=self.temporal_direction,
        )
        
        # self.lstm1 = nn.LSTM(
        #         input_size=self.heat_output_dim,
        #         hidden_size=h["lstm_hidden_dim_global"],
        #         batch_first=True,
        #         dropout=h["dropout"],
        #         bidirectional=True,
        #     )

        # self.fc_input_dim = h["lstm_hidden_dim_global"] * 2 + 768  # + 768 is for concatenation with v
        
        self.fc_input_dim = self.heat_output_dim

        self.fc1 = nn.Linear(self.fc_input_dim, self.output_dim)

class HeatLSTMNoBDR(torch.nn.Module):
    """
    Remove bidirectionality, so HEAT operates only on previous hidden states
    of an LSTM.
    
    (No) Concatenation of present LSTM. 
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length
        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)
        
        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence
            
        # Apply heat on timeline features
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction="past",
            concat_present=False,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=False,  # Not a BiLSTM input. Is LSTM.
        )
                
        # Apply the scaling
        scaler = nn.Tanh()
        h = scaler(h_heat)
        
        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h)
    
        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params
        
        self.temporal_direction = "past"

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        self.concat_present = False

        # Fix input dim and linear layer
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=768,
            present_dim=768,  
            concat_present=False,
            temporal_direction=self.temporal_direction,
        )
        
        self.lstm1 = nn.LSTM(
                input_size=768,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=False,
            )

        self.fc_input_dim = h["lstm_hidden_dim_global"] # + 768  # + 768 is for concatenation with v

        self.fc1 = nn.Linear(self.fc_input_dim, self.output_dim)

class HeatLSTMNoBDRConcatV(torch.nn.Module):
    """
    Remove bidirectionality, so HEAT operates only on previous hidden states
    of an LSTM.
    
    Concatenation of present LSTM. 
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length
        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)
        
        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence
    
        # Apply heat on timeline features
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction="past",
            concat_present=True,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=False,  # Not a BiLSTM input. Is LSTM.
        )
                
        # Apply the scaling
        scaler = nn.Tanh()
        h = scaler(h_heat)
        
        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h)
    
        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params
        
        self.temporal_direction = "both"

        # Initialize learnable HEAT parameters
        self.epsilon = nn.Parameter(
            data=torch.Tensor([h["epsilon_prior"]]), requires_grad=True
        )
        self.beta = nn.Parameter(
            data=torch.Tensor([h["beta_prior"]]), requires_grad=True
        )

        self.concat_present = True

        # Fix input dim and linear layer
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=768,
            present_dim=768,  
            concat_present=True,
            temporal_direction=self.temporal_direction,
        )
        
        self.lstm1 = nn.LSTM(
                input_size=768,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                dropout=h["dropout"],
                bidirectional=False,
            )

        self.fc_input_dim = h["lstm_hidden_dim_global"] * 2  # * 2 is for concatenation with v

        self.fc1 = nn.Linear(self.fc_input_dim, self.output_dim)
        
class LinguisticBiLSTMBiHeat(torch.nn.Module):
    """
    Uses linguistic emotion features as input to HEAT. Consists of sentiment, joy, anger, sadness, optmism scores only.
    """

    def __init__(
        self,
        input_dim=5 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)
        
        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            print("h.keys()", h.keys())
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)
        

class LinguisticConcatBiLSTMBiHeat(torch.nn.Module):
    """
    Uses linguistic emotion features as input to HEAT. Consists of sentiment, joy, anger, sadness, optmism scores only.
    """

    def __init__(
        self,
        input_dim=768+5 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)
    
        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            print("h.keys()", h.keys())
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)

# ====== Post submission 1 models ======        
# HeatBackgroundIntensityMeanBilstm, HeatDeltaReps, HeatNoMax, HeatRayleigh, HeatSoftplus
# ==========
class HeatBackgroundIntensityMeanBilstm(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
            allow_negative_decay=True,
            background_intensity="mean_lstm",
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)

class HeatDeltaReps(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
            delta_representations=True,
            allow_negative_decay=True
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)
    
class HeatNoMax(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)
        
class HeatRayleigh(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)

class HeatSoftplus(torch.nn.Module):
    """
    `bilstm_heat_concat_present_single_layer` is the
    corresponding key in the `model_selector.py` script.
    """

    def __init__(
        self,
        input_dim=768 + 1,
        output_dim=3,
        epsilon_prior=0.01,
        hidden_dim=128,
        beta_prior=0.001,
        exclude_current_post=False,
        temporal_direction="both",
        remove_padding=True,
        put_back_padding=False,
        padding_value=-123.0,
        max_seq_length=124,
        concat_present=False,
    ):
        super().__init__()
        input_dim -= 1  # Remove time from input dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heat_output_dim = return_dimensions_for_concat_heat_representations(
            input_dim=hidden_dim,
            concat_present=concat_present,
            temporal_direction=temporal_direction,
        )
        self.output_dim = output_dim

        # Define topography
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(self.heat_output_dim, output_dim)

        # HEAT parameters
        self.epsilon = epsilon_prior
        self.beta = beta_prior
        self.exclude_current_post = exclude_current_post
        self.temporal_direction = temporal_direction
        self.remove_padding = remove_padding
        self.put_back_padding = put_back_padding
        self.padding_value = padding_value
        self.max_seq_length = max_seq_length

        self.concat_present = concat_present
        self.temporal_direction = temporal_direction

    def forward(self, x):

        # Seperate auxiliary timestamp features from input features
        x, t = seperate_time_from_features(x, time_last_column=True)

        # Pre-processing - pack padding
        padded_x = handle_padding(x, to_pack=True)  # pack padded sequence

        # Get LSTM hidden states (still padded)
        lstm1_out, _ = self.lstm1(padded_x)

        # print("lstm1_out.shape = ", lstm1_out.shape)

        # Remove padding from LSTM hidden states
        lstm1_out_unpacked = handle_padding(
            lstm1_out, to_pack=False
        )  # pad packed sequence

        # Apply heat on timeline features (unpadded LSTM hidden states)
        h_heat = heat_concatenated_past_present_future(
            lstm1_out_unpacked,
            t,
            epsilon=self.epsilon,
            beta=self.beta,
            present_features=lstm1_out_unpacked,
            temporal_direction=self.temporal_direction,
            concat_present=self.concat_present,
            exclude_current_post=self.exclude_current_post,
            remove_padding=self.remove_padding,  # Remove any padding, if desired
            put_back_padding=self.put_back_padding,
            padding_value=self.padding_value,
            max_seq_length=self.max_seq_length,
            bidirectional=True,
        )

        # Select past, present, and future representations
        raw_post_embedding_dimensions = lstm1_out_unpacked.shape[-1]
        h_dim = (h_heat.shape[-1] - raw_post_embedding_dimensions) // 2
        h_past = h_heat[:, :, :h_dim]
        h_present = h_heat[:, :, h_dim:-h_dim]
        h_future = h_heat[:, :, -h_dim:]

        # Remove padding from HEAT hidden states, from dim 1
        n_posts = lstm1_out_unpacked.shape[1]
        h_past = h_past[:, :n_posts, :]
        h_present = h_present[:, :n_posts, :]
        h_future = h_future[:, :n_posts, :]

        # Apply the scaling
        scaler = nn.Tanh()
        h_past = scaler(h_past)
        h_present = scaler(h_present)
        h_future = scaler(h_future)

        # Concatenate the representations
        h_heat = torch.concat(
            [h_past, h_present, h_future], dim=-1
        )  # Concatenate along embedding dimension

        # Pass concatenated representation through the final fully connected layer
        fc1_out = self.fc1(h_heat)

        predicted_moc_encoding = F.log_softmax(fc1_out, dim=1)

        return predicted_moc_encoding

    def apply_hyperparameters(self):
        h = self.hyper_params

        # Set global hidden dimension to all lstms, if desired
        if "lstm_hidden_dim_global" in h.keys():
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm_hidden_dim_global"],
                batch_first=True,
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm_hidden_dim_global"],
                present_dim=h["lstm_hidden_dim_global"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )
        else:
            self.lstm1 = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=h["lstm1_hidden_dim"],
                batch_first=True,
                # dropout=h['dropout']
                bidirectional=True,
            )

            # Fix input dim and linear layer
            self.heat_output_dim = return_dimensions_for_concat_heat_representations(
                input_dim=h["lstm1_hidden_dim"],
                present_dim=h["lstm1_hidden_dim"] * 2,  # as bidirectional
                concat_present=self.concat_present,
                temporal_direction=self.temporal_direction,
            )

        self.fc1 = nn.Linear(self.heat_output_dim, self.output_dim)


def get_default_hyper_params(dict_hyper_params, which="learning_rate"):
    """
    Returns the specified hyperparameters, and if None are returned,
    then returns from a default list.
    """
    h = dict_hyper_params.get(which)

    if h == None:
        if which == "learning_rate":
            h = 0.01
        elif which == "epochs":
            h = 1
        elif which == "dropout":
            h = 0  # Does not apply dropout, by default

    return h
