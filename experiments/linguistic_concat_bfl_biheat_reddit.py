"""
BiHEAT, but only on linguistic features:
sentiment, joy, sadness, anger, optimsm.
"""

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
    model_names=["linguistic_concat_bilstm_biheat_concat_h"],
    embedding_type=None,
    features=["bert_focal_loss_emotion"],
    which_loss="cross_entropy",
    datasets=["reddit"],
    hyperparams_to_search=model_hyperparameters_to_search_global,
    experiment_name=return_file_name(os.path.basename(__file__)),
    prototype=False,
    verbose=True,
    display_progress_bar=True,
)
