import os
import sys

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
from pipelines.full_pipeline import (
    full_pipeline_for_multiple_models,
    full_pipeline_for_single_model,
)

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from export_data import export_experiment_results, return_file_name
from notification_bot import send_email
from verbose import compute_number_of_iterations

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# from verbose import progress_bar


def main():
    prototype = False
    verbose = False
    # embedding_type = "bert_focal_loss"
    embedding_type = (
        "sentence-bert"  # Try again with sentence-bert, to see that results are decent
    )
    experiment_name = return_file_name(
        os.path.basename(__file__)
    )  # datetime_lstm_variants
    datasets = ["reddit"]

    config = {}
    config["loss_function_type"] = "class_balanced_focal_loss"
    config["gamma"] = 2.0  # Hyper-parameter for focal loss
    config[
        "beta_class_balanced"
    ] = -0.9999  # Hyper-parameter for class balanced focal loss

    model_names = [
        # LSTM: Concat, but exclude current post in HEAT calculation
        # "lstm_vanilla",
        "bilstm"
        # "lstm_concat_exclude_present_in_heat_past_present_future",
        # "lstm_vanilla",
        # "bilstm",
    ]

    hyperparams_to_search = {
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
    }
    config["hyper_params_to_search"] = hyperparams_to_search
    config["embedding_type"] = embedding_type

    display_progress_bar = True
    number_of_iterations = compute_number_of_iterations(
        hyperparams_to_search, datasets, model_names
    )

    with alive_bar(number_of_iterations) as progress_bar:
        for i, which_data in enumerate(
            datasets
        ):  # Carry out pipeline, for multiple models

            #   if display_progress_bar:
            config["which_dataset"] = which_data
            config["which_dataset_i"] = i
            config["len_datasets"] = len(datasets)
            config["len_models"] = len(model_names)

            classification_reports, results = full_pipeline_for_multiple_models(
                progress_bar=progress_bar,
                model_names=model_names,
                which_data=which_data,
                prototype=prototype,
                hyperparams_to_search=hyperparams_to_search,
                features=[embedding_type],
                target="label_3",
                perform_early_stopping=True,
                verbose=verbose,
                experiment_name=experiment_name,
                early_stopping_criterion="macro_f1",
                display_progress_bar=display_progress_bar,
                config=config,
            )

            export_experiment_results(
                classification_reports,
                results,
                config,
                experiment_name=experiment_name,
                dataset=which_data,
                folder="results",
            )


if __name__ == "__main__":

    main()
    # try:
    #     main()
    # except Exception as error_message:
    #     send_email(
    #         subject="[{}] Error, script failed.".format(os.path.basename(__file__)),
    #         message=str(error_message),
    #         receiving_email="angryasparagus@hotmail.com",
    #     )
