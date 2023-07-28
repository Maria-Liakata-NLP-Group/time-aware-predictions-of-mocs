import os
import sys

sys.path.insert(
    0, "../../predicting_mocs/"
)  # Adds higher directory to python modules path
from utils.run_experiment import run_experiment
from global_parameters import model_hyperparameters_to_search_global

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from export_data import return_file_name

run_experiment(
    model_names=["2_layer_bilstm_heat_concat_bilstm_with_linear_layer_tanh"],
    embedding_type="bert_focal_loss",
    which_loss="cross_entropy",
    datasets=["reddit"],
    hyperparams_to_search=model_hyperparameters_to_search_global,
    experiment_name=return_file_name(os.path.basename(__file__)),
    prototype=False,
    verbose=True,
    display_progress_bar=True,
)
