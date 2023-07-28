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
from global_parameters import talklife_model_hyperparameters_to_search_global

run_experiment(
    model_names=["bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh"],
    embedding_type="bert_focal_loss",
    which_loss="cross_entropy",
    datasets=["talklife"],
    hyperparams_to_search=talklife_model_hyperparameters_to_search_global,
    experiment_name=return_file_name(os.path.basename(__file__)),
    prototype=False,
    verbose=True,
    display_progress_bar=True,
    folds=[0],
    random_seed=2,
)
