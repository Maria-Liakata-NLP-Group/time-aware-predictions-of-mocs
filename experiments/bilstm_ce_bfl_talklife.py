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


run_experiment(
    model_names=["bilstm"],
    embedding_type="bert_focal_loss",
    which_loss="cross_entropy",
    datasets=["talklife"],
    hyperparams_to_search={
        "learning_rate": [0.0001, 0.001],
        "epochs": [100],
        "epsilon_prior": [0.001, 0.01, 0.1],
        "beta_prior": [0.001, 0.01, 0.1],
        "dropout": [0, 0.25, 0.50],
        "lstm_hidden_dim_global": [128, 256, 512],
        "batch_size": [1],
        "patience": [5],
        "gamma": [2.0],
        "beta_cb": [0.9, 0.8, 0.7, 0.6],
    },
    experiment_name=return_file_name(os.path.basename(__file__)),
    prototype=False,
    verbose=False,
    display_progress_bar=True,
)
