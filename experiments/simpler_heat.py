import os
import sys

sys.path.insert(
    0, "../../predicting_mocs/"
)  # Adds higher directory to python modules path
from utils.run_experiment import run_experiment

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from export_data import return_file_name
from global_parameters import model_hyperparameters_to_search_global

run_experiment(
    model_names=[
        "bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh_allow_negative_decayed_x_no_summation",
        # "learnable_heat_softplus_allow_negative_no_summation_lstm_heat_concat_single_layer_with_linear_layer_tanh",
        # "bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh_no_summation",
        # "bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh_allow_negative_decayed_x",
    ],
    embedding_type="bert_focal_loss",
    which_loss="cross_entropy",
    datasets=["reddit"],
    hyperparams_to_search=model_hyperparameters_to_search_global,
    experiment_name=return_file_name(os.path.basename(__file__)),
    prototype=False,
    verbose=True,
    display_progress_bar=True,
)
