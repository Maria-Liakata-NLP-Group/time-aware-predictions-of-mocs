import os
import sys
import traceback

import torch
from alive_progress import alive_bar

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path
from utils.io import data_handler
from utils.io.my_pickler import my_pickler

sys.path.insert(
    0, "../../predicting_mocs/"
)  # Adds higher directory to python modules path
from models import heat
from pipelines.full_pipeline import full_pipeline_for_multiple_models

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from global_parameters import device_to_use
from export_data import export_experiment_results, return_file_name
from notification_bot import send_email
from verbose import compute_number_of_iterations

device = torch.device(device_to_use if torch.cuda.is_available() else "cpu")
print(device)

experiment_name = return_file_name(os.path.basename(__file__))  # datetime_lstm_variants


def run_experiment(
    model_names=["bilstm"],
    hyperparams_to_search={
        "learning_rate": [0.0001, 0.001],
        "epochs": [100],
        "epsilon_prior": [0.001, 0.01, 0.1],
        "beta_prior": [0.001, 0.01, 0.1],
        "dropout": [0, 0.25, 0.50],
        # "lstm1_hidden_dim": [64,128,256, 512],
        # "lstm2_hidden_dim": [64,128,256, 512],
        "lstm_hidden_dim_global": [128, 256, 512],
        "batch_size": [1],
        # "number_of_lstm_layers": [2],
        "patience": [5],
        "gamma": [2.0],
        "beta_cb": [-0.9999, -0.999, -0.99, -0.9],
    },
    experiment_name="",
    prototype=False,
    verbose=False,
    display_progress_bar=True,
    embedding_type="sentence-bert",
    features=[],
    datasets=["reddit", "talklife"],
    which_loss="cross_entropy",
    early_stopping_criterion="loss",
    folds='all',
    random_seed=0
):

    if verbose:
        print("VERBOSE == TRUE")
    try:
        # Store config information
        config = {}
        config["loss_function_type"] = which_loss
        config["hyper_params_to_search"] = hyperparams_to_search
        config["embedding_type"] = embedding_type
        config["model_names"] = model_names
        config["prototype"] = prototype
        config["features"] = features
        config["random_seed"] = random_seed

        display_progress_bar = True
        number_of_iterations = compute_number_of_iterations(
            hyperparams_to_search, datasets, model_names, loss_name=which_loss, folds=folds
        )

        with alive_bar(
            number_of_iterations, title="Running full experiment..."
        ) as progress_bar:
            for i, which_data in enumerate(
                datasets
            ):  # Carry out pipeline, for multiple models

                #   if display_progress_bar:
                config["which_dataset"] = which_data
                config["which_dataset_i"] = i
                config["len_datasets"] = len(datasets)
                config["len_models"] = len(model_names)
                config["folds"] = folds
                
                all_features = []
                if embedding_type != None:
                    all_features.append(embedding_type)
                all_features.extend(features)  # Add remaining features (e.g. linguistic ones)

                classification_reports, results = full_pipeline_for_multiple_models(
                    progress_bar=progress_bar,
                    model_names=model_names,
                    which_data=which_data,
                    prototype=prototype,
                    hyperparams_to_search=hyperparams_to_search,
                    features=all_features,
                    target="label_3",
                    perform_early_stopping=True,
                    early_stopping_criterion=early_stopping_criterion,
                    verbose=verbose,
                    experiment_name=experiment_name,
                    display_progress_bar=display_progress_bar,
                    config=config,
                    folds=folds,
                )

                export_experiment_results(
                    classification_reports,
                    results,
                    # config,
                    experiment_name=experiment_name,
                    dataset=which_data,
                    folder="results",
                    folds=folds
                )
    except Exception:
        send_email(
            subject="[{}] Error, script failed.".format(experiment_name),
            message="Error:\n\n" + str(traceback.format_exc()),
            receiving_email="angryasparagus@hotmail.com",
        )
