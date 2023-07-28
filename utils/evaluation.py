import sys

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

from utils.io import data_handler

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from global_parameters import device_to_use
device = torch.device(device_to_use if torch.cuda.is_available() else "cpu")



# def aggregate_results_across_folds(
#     df_all_results, y_true, y_pred, fold=0, reset_index=True
# ):
#     """
#     Stores results over different folds into a single DataFrame, which will
#     then be used to evaluate the models across the whole dataset.
#     """
#     # Move to CPU, so place in Pandas Series
#     y_true = y_true.clone().cpu()
#     y_pred = y_pred.clone().cpu()

#     # Convert one-hot arrays to 1D series, if they are multi-dimensional
#     if y_true.ndim > 1:
#         y_true = data_handler.convert_one_hot_array_back_to_labels_df(
#             y_true, reverse_one_hot=True
#         )
#     if y_pred.ndim > 1:
#         y_pred = data_handler.convert_one_hot_array_back_to_labels_df(
#             y_pred, reverse_one_hot=True
#         )

#     # Convert to Series, if not already:
#     if not isinstance(y_true, pd.Series):
#         y_true = pd.Series(y_true)
#     if not isinstance(y_pred, pd.Series):
#         y_pred = pd.Series(y_pred)

#     # Create DataFrame for current set of results
#     df_current_results = pd.DataFrame()
#     df_current_results["y_true"] = y_true
#     df_current_results["y_pred"] = y_pred
#     # df_current_results["post_index"] = post_index
#     df_current_results["fold"] = fold

#     # Aggregate to full DataFrame, across all folds
#     if len(df_all_results) == 0:
#         df_all_results = df_current_results
#     else:
#         df_all_results = pd.concat([df_all_results, df_current_results], axis=0)

#     # Reset the index
#     if reset_index:
#         df_all_results = df_all_results.reset_index().drop("index", axis=1)

#     return df_all_results


def evaluate_model(
    model,
    test_loader,
    metric="macro_f1",
    verbose=True,
    remove_padding=False,
    loss_fn=nn.CrossEntropyLoss(),
    padding_value=-123.0,
    n_classes=3,
):
    """
    Evaluates a model on a given hold-out dataloder. Will not update model
    architecture, and is done only to get metric score.
    """
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for _, (inputs, y_true, post_id) in enumerate(test_loader):

            # Move to GPU if available
            inputs, y_true, post_id = inputs.to(device), y_true.to(device), post_id.to(device)

            y_pred = model(inputs)

            # Unroll Tensors to a scalar 1D array of label-encoded predictions: Output is (n_samples)
            y_true, y_pred, post_id = return_unrolled_y_true_and_y_pred(
                inputs,
                y_true,
                y_pred,
                post_id,
                remove_padding=remove_padding,
                padding_value=padding_value,
                n_classes=n_classes,
                max_seq_length=124,
            )

            # print("y_pred.shape: ", y_pred.shape)

            # Store all predictions and true values, for all samples
            y_preds.append(y_pred)
            y_trues.append(y_true)

        actuals, predictions = torch.cat(y_trues), torch.cat(y_preds)

    model.train()  # Unfreeze weights

    # Evaluate how good the predictions align with the true values
    actuals = actuals.clone().cpu()  # Move to GPU
    predictions = predictions.clone().cpu()
    score = evaluate_predictions(
        y_true=actuals, y_pred=predictions, metric=metric, loss_fn=loss_fn
    )
    # if verbose:
    #     print("{}: {:.4f}".format(metric, score))

    return score


def evaluate_loss(
    model,
    dataloader,
    verbose=True,
    epochs=1,
    remove_padding=False,
    loss_fn=nn.CrossEntropyLoss(),
    n_classes=3,
    padding_value=-123.0,
    return_as_average=True,
):
    """
    Ensure no steps are taken in optimization, so model is not exposed to validation set.

    Sums the loss, and returns it. Should average it by the length of the dataloader.
    """

    # print("`evaluate_loss` called.")

    model.eval()  # Freeze weights
    loss_for_current_epoch = 0.0
    number_of_samples = 0
    with torch.no_grad():  # Not sure if necessary, as I already set model.eval()
        for i, train_data in enumerate(dataloader, 0):  # Loop over samples (batches)
            inputs, y_true, post_id = train_data

            # Move to GPU
            inputs, y_true, post_id = inputs.to(device), y_true.to(device), post_id.to(device)

            # Forward
            y_pred = model(inputs)

            # print("(before pad removed) y_pred.shape: ", y_pred.shape)

            # Unroll Tensors to a scalar 1D array of label-encoded predictions: Output is (n_samples)
            y_true, y_pred, post_id = return_unrolled_y_true_and_y_pred(
                inputs,
                y_true,
                y_pred,
                post_id,
                remove_padding=remove_padding,
                padding_value=padding_value,
                n_classes=n_classes,
                max_seq_length=124,
                retain_predictions_as_probabilities=True,
            )

            # Evaluate loss
            loss = loss_fn(y_pred, y_true)

            # Store running loss, for simply printing statistics
            loss_for_current_epoch += loss.item()

            number_of_samples += y_pred.shape[
                0
            ]  # Number of samples across all batches. y_pred is unpadded

    loss = loss_for_current_epoch
    average_loss = loss / number_of_samples
    model.train()  # Unfreeze weights

    if return_as_average:
        return average_loss
    else:
        return loss


# def return_true_and_predicted_values_from_model_and_dataloader(
#     model, test_loader, remove_padding=False, padding_value=-123.0, n_classes=3
# ):
#     """
#     Returns the true values and the predicted values by an input model for a
#     given data_loader. Ensures the model does not get trained.

#     TODO: Note that when batch size > 1, these values are actually padded to
#     size of the longest timeline in the batch! As they are an array. Must ensure
#     that these padded values are removed before evaluation.
#     """
#     model.eval()
#     y_preds, y_trues = [], []
#     with torch.no_grad():
#         for _, (inputs, y_true, post_id) in enumerate(test_loader):

#             inputs, y_true = inputs.to(device), y_true.to(device)
#             y_pred = model(inputs)

#             # Unroll Tensors to a scalar 1D array of label-encoded predictions: Output is (n_samples)
#             y_true, y_pred, post_id = return_unrolled_y_true_and_y_pred(
#                 inputs,
#                 y_true,
#                 y_pred,
#                 post_id,
#                 remove_padding=remove_padding,
#                 padding_value=padding_value,
#                 n_classes=n_classes,
#                 max_seq_length=124,
#             )

#             # Store all predictions and true values, for all samples
#             y_preds.append(y_pred)
#             y_trues.append(y_true)

#         actuals, predictions = torch.cat(y_trues), torch.cat(y_preds)
#     model.train()  # Unfreeze weights

#     return actuals, predictions

def return_true_and_predicted_values_from_model_and_dataloader(
    model, test_loader, remove_padding=False, padding_value=-123.0, n_classes=3
):
    """
    Returns the true values and the predicted values by an input model for a
    given data_loader. Ensures the model does not get trained.
    """
    model.eval()
    y_preds, y_trues, post_ids = [], [], []
    with torch.no_grad():
        for _, (inputs, y_true, post_id) in enumerate(test_loader):

            inputs, y_true, post_id = inputs.to(device), y_true.to(device), post_id.to(device)
            y_pred = model(inputs)

            # Unroll Tensors to a scalar 1D array of label-encoded predictions: Output is (n_samples)
            y_true, y_pred, post_id = return_unrolled_y_true_and_y_pred(
                inputs,
                y_true,
                y_pred,
                post_id,
                remove_padding=remove_padding,
                padding_value=padding_value,
                n_classes=n_classes,
                max_seq_length=124,
            )

            # Store all predictions and true values, for all samples
            y_preds.append(y_pred)
            y_trues.append(y_true)
            post_ids.append(post_id)

        actuals, predictions, pids = torch.cat(y_trues), torch.cat(y_preds), torch.cat(post_ids)
    model.train()  # Unfreeze weights

    return actuals, predictions, pids


# def return_unrolled_y_true_and_y_pred(
#     inputs,
#     y_true,
#     y_pred,
#     remove_padding=True,
#     padding_value=-123.0,
#     n_classes=3,
#     max_seq_length=124,
#     retain_predictions_as_probabilities=False,
# ):
#     """
#     Returns a scalar (1D) tensor of the true and predicted values for a given set of inputs from a dataloader.
#     Ensures the model does not get trained, and that the padding is removed.

#     Args:
#         inputs (_type_): _description_
#         y_true (_type_): _description_
#         model (_type_): _description_
#         remove_padding (bool, optional): _description_. Defaults to False.
#         padding_value (float, optional): _description_. Defaults to -123.0.
#         n_classes (int, optional): _description_. Defaults to 3.
#         retain_predictions_as_probabilities (bool, optional): _description_. Defaults to False. If True, the predictions are returned as probabilities, rather than label-encoded.

#     Returns:
#         _type_: _description_
#     """
#     # For padding removal
#     is_not_padding = inputs != padding_value  # Mask: True if not padding
#     is_not_padding = is_not_padding[:, :, 0]  # Remove embedding dimension
#     is_not_padding = is_not_padding.view(-1)  # Unroll the timeline dimension
#     # sequence_length = is_not_padding.sum(axis=1)

#     # Re-apply padding to y_pred, so is consistent shape with y_true
#     # TODO: Perhaps a better solution is to ensure that the model outputs are always fully padded, and then remove the padding here. The model could also output padded mask
#     y_pred = data_handler.pad_batched_tensor(
#         y_pred,
#         max_seq_length=max_seq_length,
#         padding_value=padding_value,
#     )

#     # Unroll the batch dimension
#     y_pred = y_pred.view(-1, n_classes)
#     y_true = y_true.view(-1, n_classes)

#     # Label encoding
#     if (
#         not retain_predictions_as_probabilities
#     ):  # For loss functions, this is not used - as they can handle probabilities
#         y_pred = torch.argmax(
#             y_pred, dim=1
#         )  # Convert predictions to label-encoding, if desired
#     y_true = torch.argmax(y_true, dim=1)

#     # Remove padding
#     y_pred = y_pred.to(device)  # Move to GPU
#     is_not_padding = is_not_padding.to(device)  # Move to GPU
#     y_true = y_true[is_not_padding]
#     y_pred = y_pred[is_not_padding]


#     return y_true, y_pred

def return_unrolled_y_true_and_y_pred(
    inputs,
    y_true,
    y_pred,
    post_ids,
    remove_padding=True,
    padding_value=-123.0,
    n_classes=3,
    max_seq_length=124,
    retain_predictions_as_probabilities=False,
):
    """
    Returns a scalar (1D) tensor of the true and predicted values for a given set of inputs from a dataloader.
    Ensures the model does not get trained, and that the padding is removed.

    Args:
        inputs (_type_): _description_
        y_true (_type_): _description_
        model (_type_): _description_
        remove_padding (bool, optional): _description_. Defaults to False.
        padding_value (float, optional): _description_. Defaults to -123.0.
        n_classes (int, optional): _description_. Defaults to 3.
        retain_predictions_as_probabilities (bool, optional): _description_. Defaults to False. If True, the predictions are returned as probabilities, rather than label-encoded.

    Returns:
        _type_: _description_
    """
    # For padding removal
    is_not_padding = inputs != padding_value  # Mask: True if not padding
    is_not_padding = is_not_padding[:, :, 0]  # Remove embedding dimension
    is_not_padding = is_not_padding.view(-1)  # Unroll the timeline dimension

    # Re-apply padding to y_pred, so is consistent shape with y_true
    # TODO: Perhaps a better solution is to ensure that the model outputs are always fully padded, and then remove the padding here. The model could also output padded mask
    y_pred = data_handler.pad_batched_tensor(
        y_pred,
        max_seq_length=max_seq_length,
        padding_value=padding_value,
    )

    # Unroll the batch dimension
    y_pred = y_pred.view(-1, n_classes)
    y_true = y_true.view(-1, n_classes)
    post_ids = post_ids.view(-1)
    
    # Label encoding
    if (
        not retain_predictions_as_probabilities
    ):  # For loss functions, this is not used - as they can handle probabilities
        y_pred = torch.argmax(
            y_pred, dim=1
        )  # Convert predictions to label-encoding, if desired
    y_true = torch.argmax(y_true, dim=1)

    # Remove padding
    y_pred = y_pred.to(device)  # Move to GPU
    is_not_padding = is_not_padding.to(device)  # Move to GPU
    y_true = y_true[is_not_padding]
    y_pred = y_pred[is_not_padding]
    post_ids = post_ids[is_not_padding]


    return y_true, y_pred, post_ids

def print_metrics(n_correct, n_samples, test_loss, metrics, to_return, verbose=True):
    """
    Prints out metrics based on the number of true positives on the test set.
    """
    string_to_print = ""
    scores = {}
    for metric in metrics:
        if metric == "accuracy":
            scores[metric] = 100 * n_correct / n_samples
            string_to_print += "Accuracy: {:.4f} | \t".format(scores[metric])
        if metric == "loss":
            scores[metric] = test_loss
            string_to_print += "Average Loss: {:.4f} | \t".format(scores[metric])

    if verbose:
        print(string_to_print)

    if to_return != None:
        return scores[metric]


def train_validate(model, train_loader, val_loader, n_epochs, verbose=True):
    if verbose:
        print("Training and validating for {} epochs".format(n_epochs))

    for e in range(epochs):
        if verbose:
            print(f"Epoch {e+1}\n-------------------------------")
            training_loop(train_loader, model, loss_fn, optimizer)
            testing_loop(val_loader, model, loss_fn)
        if verbose:
            print("Finished training and validating!")

    # elif metric == "macro_f1":
    #     score = f1_score(actuals, predictions, average="macro")


def evaluate_predictions(
    y_true, y_pred, metric="accuracy", zero_division=0, loss_fn=nn.CrossEntropyLoss()
):

    # Convert to CPU, so sklearn can handle
    # y_true = y_true.clone().cpu()
    # y_pred = y_pred.clone().cpu()

    if metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    elif metric == "macro_f1":
        score = f1_score(y_true, y_pred, average="macro")
    elif metric == "precision":
        score = precision_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        )
    elif metric == "recall":
        score = recall_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        )
    elif metric == "loss":
        # loss_fn = nn.CrossEntropyLoss()
        print(y_pred.shape)
        print()
        print(y_true.shape)
        print(y_pred.dtype())
        print(y_true.dtype())

        # .float()
        score = loss_fn(y_pred, y_true)

    return score


def return_confusion_matrix_of_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    return cm


def return_classification_report(
    y_true,
    y_pred,
    target_names=["S", "E", "O"],
    zero_division=0,
    output_dict=True,
    as_dataframe=True,
):
    if as_dataframe:
        output_dict = True

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=zero_division,
        output_dict=output_dict,
        labels=[0, 1, 2]
    )

    if as_dataframe:
        report = pd.DataFrame(report)

    return report


def visualize_classification_report_for_method_using_dict_report(
    dict_report,
    model_name="method_name",
    metrics=["precision", "recall", "f1-score"],
    target_names=["S", "E", "O"],
):
    """
    Returns a table just like as is presented in the CLPsych overview paper,
    for a given dictionary.
    """
    # Select just precision, recall, f1 for each class and macro average
    columns_to_retain = ["macro avg"] + target_names
    df = pd.DataFrame(dict_report)[columns_to_retain]
    df = df.loc[metrics]

    # Rename to P, R, F1
    df = df.rename(index={"precision": "P", "recall": "R", "f1-score": "F1"})

    # Convert to multi-level index DataFrame, where row index shows model name
    df = pd.concat(dict(row_index=df), axis=0)
    df = df.rename({"row_index": model_name})

    # Transpose the multi-row index to be a multi-column index
    df = df.unstack()

    # Reorder the multi-column index
    cols = ["P", "R", "F1"]
    new_cols = df.columns.reindex(cols, level=1)
    df = df.reindex(columns=new_cols[0])

    return df


def classification_report_for_single_method_using_y(
    y_true,
    y_pred,
    model_name="model_name",
    target_names=["S", "E", "O"],
    zero_division=0,
    metrics=["precision", "recall", "f1-score"],
):
    """
    Extremely useful function.

    Returns a table just like as is presented in the CLPsych overview paper,
    for a set of true values and predicted values. The row index is given by
    `model_name`.
    """

    dict_report = return_classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=zero_division,
        output_dict=True,
        as_dataframe=False,
    )

    df = visualize_classification_report_for_method_using_dict_report(
        dict_report, model_name=model_name, metrics=metrics, target_names=target_names
    )

    return df


def concatenate_classification_reports_for_multiple_methods(single_report, all_reports):
    """
    Takes an input DataFrame, containing results for all the reports, and the
    report for a single method. Concatenates them and returns the concatenated
    version.
    """
    if len(all_reports) > 0:
        all_reports = pd.concat([all_reports, single_report], axis=0)
    else:
        all_reports = single_report

    return all_reports


def sort_classification_reports(reports, by=[("macro avg", "F1")], ascending=False):
    """
    Sorts the classification report by (default: macro avg F1 score).
    """

    sorted_reports = reports.sort_values(by=by, ascending=ascending)

    return sorted_reports


def evaluate_df(df, metric="accuracy"):
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    score = evaluate_predictions(y_true, y_pred, metric)

    return score


def train_evaluate_return_dataframe_of_metrics(
    dataset, models=[], metrics=["precision", "recall", "f1"]
):
    """
    Returns a DataFrame containing the final scores of the different models,
    for several metrics. Precision, Recall, F1 - for macro-average and per
    class.
    """

    return None


def window_based_timeline_evaluation(
    y_true,
    time_true,
    y_pred,
    time_pred,
    metrics=["precision", "recall", "F1"],
    method_name="method_name",
    tau=5,
):
    """
    Contains 4 sequences, and returns the metrics (y-axis) as a graph, for varying
    window sizes (x-axis) for a given method.

        y_true (3d Tensor): True values of the MoCs (S, E, O)

        time_true (1d Tensor): Timestamps (epoch time, days) of the associated
        true labels

        y_pred (3d Tensor): Predicted labels of MoCs (S, E, O)

        time_pred (1d Tensor): Associated timestamps (epoch time, days) of the
        predicted labels

        tau (int/ float): The threshold margin of error (in days) for which a
        predicted MoC is classified as a true positive against y_true.
    """

    # Check if y_pred falls in window tau of y_true

    return None


# def class_balanced_focal_loss():

# def save_model_results(y_true, y_pred, model_name="logreg", dataset="talklife"):

#     return None


# def predict_on_test_set(model, dataloader, optimizer):

#     all_y_pred = None
#     for _, data in enumerate(dataloader):

#         # Every data instance is an input + label pair
#         inputs, labels = data

#         optimizer.zero_grad()

#         y_pred = model(inputs)
#         all_y_pred = data_handler.aggregate_over_loop(all_y_pred, y_pred)

#     return all_y_pred


# def testing_loop(
#     model,
#     test_loader,
#     loss_fn,
#     verbose=True,
#     metrics=["accuracy", "loss"],
#     to_return=None,
# ):
#     """
#     Returns the accuracy and loss of an input model on an input test loader.
#     Does not train the model.
#     """
#     n_samples = len(test_loader.dataset)
#     num_batches = len(test_loader)
#     test_loss, n_true_positives = 0, 0

#     with torch.no_grad():  # Don't make changes to the model
#         for x, y in test_loader:
#             y_pred = model(x)
#             test_loss += loss_fn(y_pred, y).item()
#             n_true_positives += (
#                 (y_pred.argmax(1) == y).type(torch.float).sum().item()
#             )  # Returns +1 if matches true

#     test_loss /= num_batches

#     score = print_metrics(
#         n_correct=n_true_positives,
#         n_samples=n_samples,
#         test_loss=test_loss,
#         metrics=metrics,
#         to_return=to_return,
#         verbose=verbose,
#     )

#     return score


def predict_with_stored_model(model_name, dataset, model_dir="models/"):
    """
    Loads a model from the model directory, and predicts on the test set of the
    input dataset.
    """

    # Load the model
    model = load_model(model_name, dataset, model_dir=model_dir)

    # Load the test set
    test_loader = load_dataset(dataset, "test")

    # Predict on the test set
    y_pred = predict_on_test_set(model, test_loader)

    return y_pred