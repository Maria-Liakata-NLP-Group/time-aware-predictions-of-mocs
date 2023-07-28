import pandas as pd
import torch

import sys

sys.path.insert(
    0, "../../predicting_mocs/"
)  # Adds higher directory to python modules path

# from utils.io import data_handler
from utils.visualize import aggregate_and_evaluate_results_from_multiple_experiments
from pipelines.full_pipeline import get_datasets
from utils.kfold import recreate_dataloaders_with_batch_size, aggregate_results_across_folds
from utils.evaluation import (
    # aggregate_results_across_folds,
    classification_report_for_single_method_using_y,
    evaluate_df,
    evaluate_model,
    return_true_and_predicted_values_from_model_and_dataloader,
)


sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path
from utils.io.data_handler import load_model, set_random_seeds
from utils.io.my_pickler import my_pickler

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from global_parameters import device_to_use

seed = 0
set_random_seeds(seed)
device = torch.device(device_to_use if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))


# Load the stored model names
file_names = [
    # socialmediaVM
    # "2023_03_28_17_00_08____tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy",
    # "2023_04_05_16_14_43____1_layer_bilstm_cross_entropy",
    
    # sanctus
    # "2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h",
    # "2023_04_15_15_35_02____reddit_heat_no_msd"
]

# Define the model name
# model_name = "2023_03_28_17_00_08____tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy"
model_name = "2023_04_05_16_14_43____1_layer_bilstm_cross_entropy"
# model_name = "2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h"
# model_name = "2023_04_15_15_35_02____reddit_heat_no_msd"



dataset_name = 'reddit'
k = 1   # default for reddit
# k = -1
embedding_type = 'bert_focal_loss'
batch_size = 32
# is_time_aware = True
is_time_aware = False
train_val_test_sizes = [3, 1, 1]
target = 'label_3'

# Load stored model
print("Loading model... : `{}`".format(model_name))
model = load_model("", model_name, file_type="torch")
model = model.to(device)

# Define the testloader, without shuffling
print("Loading dataset... : `{}`".format(dataset_name))
datasets = get_datasets(
        which_data=dataset_name,
        prototype=False,
        aggregate_to_timeline_level=True,
        apply_padding=True,
        max_seq_length=124,
        padding_value=-123.0,
        embedding_type=embedding_type,
    )
dataset = datasets[dataset_name]

# Specificy args for creating the dataloaders
data_loader_config = {
    "dataset": dataset,
    "k": k,
    "train_val_test_sizes": train_val_test_sizes,
    "features": [embedding_type],
    "target": target,
    "batch_size": batch_size,  # by default - this will be updated based on hyperparams
    "shuffle": False,
    "num_workers": 1,
    "is_time_aware": is_time_aware,
    "dataset_name": dataset_name,
}

# Create testloader
print("Creating testloader...")
(
train_loader,
val_loader,
train_and_val_loader_combined,
test_loader,
) = recreate_dataloaders_with_batch_size(
    data_loader_config,
    train_loader=None,
    val_loader=None,
    train_and_val_loader_combined=None,
    test_loader=None,
)


# Run the model on the testloader
print("Running model on testloader...")
y_true, y_pred, post_ids = return_true_and_predicted_values_from_model_and_dataloader(
        model, test_loader, remove_padding=True
    )

k = -1
# Store predictions in dataframe
print("Storing predictions in dataframe...")
df_all_test_results = pd.DataFrame()
df_all_test_results = aggregate_results_across_folds(pd.DataFrame(), y_true, y_pred, post_ids, fold=k,
)


# Join the preditions with the meta-data


# Save the joined metadata and predictions to a pickle file, for later analysis
# my_pickler("o", "error_analysis_df_all_test_results_{}".format(model_name), df_all_test_results, folder='results')
my_pickler("o", "test_error_analysis_df_all_test_results_{}".format(model_name), df_all_test_results, folder='results')

# # Generate classification report
print("Generating classification report...")
classification_report = classification_report_for_single_method_using_y(
        df_all_test_results["y_true"],
        df_all_test_results["y_pred"],
        model_name=model_name,
        target_names=["S", "E", "O"],
        zero_division=0,
        metrics=["precision", "recall", "f1-score"],
    )

print(classification_report)
print("macro avg F1", classification_report[('macro avg', 'F1')])