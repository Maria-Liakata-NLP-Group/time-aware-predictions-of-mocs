"""
Performs k-fold cross-validation on an input model
and

dataset.
"""
import itertools
import sys

import pandas as pd
import torch
from alive_progress import config_handler
from sklearn import preprocessing
from torch import nn, optim
from tqdm import tqdm

sys.path.insert(0, "../../predicting_mocs/")
from models.model_selector import (
    check_if_model_is_time_aware,
    check_if_model_is_timeline_sensitive,
    get_default_hyper_params,
    model_selector,
    return_only_valid_hyper_parameters_for_model,
)
from utils.loss_functions import create_loss_function

from utils.evaluation import (
    # aggregate_results_across_folds,
    classification_report_for_single_method_using_y,
    evaluate_df,
    evaluate_model,
    return_true_and_predicted_values_from_model_and_dataloader,
)
from utils.training import early_stopping, training_loop

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path
from utils.io import data_handler


def kfold(
    progress_bar,
    model=None,
    dataset=None,
    dataset_name=None,
    model_name="model_name",
    features=["sentence-bert"],
    target="label_3",
    train_val_test_sizes=[3, 1, 1],
    shuffle=False,
    num_workers=1,
    hyperparams_to_search={"learning_rate": [0.01, 0.1], "epochs": [1, 10]},
    metric="macro_f1",
    save_models=False,
    verbose=True,
    is_timeline_sensitive=False,
    is_time_aware=False,
    apply_standard_scaling=False,
    perform_early_stopping=True,
    early_stopping_criterion="loss",
    experiment_name="default_experiment_name",
    display_progress_bar=True,
    config={},
    folds="all",
):
    """
    Returns the train, val, test score.

    Performs K-fold cross validation on a given model, using a specified dataset
    which has the form of a dataframe.
    """
    train_scores = pd.Series(pd.Series(dtype="float64"))
    val_scores = pd.Series(pd.Series(dtype="float64"))
    all_optimal_hyperparams = []
    all_best_models = []

    total_size = len(dataset)
    df_all_test_results = pd.DataFrame()  # Initial empty df

    data_loader_config = {}

    # No need for 5 fold CV. Just have a single validation set.
    if dataset_name == "reddit":
        k_folds = [1]  # Note, for Reddit, this k is just for the validation set. Test set is always -1, hardcoded.
    else:  # Otherwise, for TalkLife, perform 5 fold CV or use the specified folds
        if folds == "all":
            k_folds = [0, 1, 2, 3, 4]
        else:
            k_folds = folds

    # Loop over each specified fold
    for k in k_folds:
        if verbose:
            print("\t[K={}/{}]\t({})".format(k, k_folds, model_name))
        if display_progress_bar:
            config["k"] = k
            config["k_folds"] = k_folds

        # )  # Recreate the model, and re-train it
        if verbose:
            print("Kfold: {} / {}".format(k, k_folds))

        # Specificy args for creating the dataloaders
        data_loader_config = {
            "dataset": dataset,
            "k": k,
            "train_val_test_sizes": train_val_test_sizes,
            "features": features,
            "target": target,
            "batch_size": -9999,  # by default - this will be updated based on hyperparams
            "shuffle": shuffle,
            "num_workers": num_workers,
            "is_time_aware": is_time_aware,
            "dataset_name": dataset_name,
        }
        
        # Select best model: grid-search to find the best model and hyperparams
        (
            model,
            train_score,
            val_score,
            optimal_hyperparams,
            test_loader,
        ) = grid_search_using_validation_set(
            progress_bar=progress_bar,
            model_name=model_name,
            hyperparams_to_search=hyperparams_to_search,
            save_model=False,
            is_timeline_sensitive=is_timeline_sensitive,
            verbose=verbose,
            apply_standard_scaling=apply_standard_scaling,
            perform_early_stopping=perform_early_stopping,
            early_stopping_criterion=early_stopping_criterion,
            experiment_name=experiment_name,
            config=config,
            display_progress_bar=display_progress_bar,
            data_loader_config=data_loader_config,
        )

        # Predict, using optimal model
        if -1 in dataset["fold"].values:
            pass
        else:
            y_true, y_pred, post_ids = return_true_and_predicted_values_from_model_and_dataloader(
                model, test_loader, remove_padding=is_timeline_sensitive
            )

        # Store scores
        train_scores.at[k] = train_score
        val_scores.at[k] = val_score

        # Aggregate results for test set over all for loops
        if -1 in dataset["fold"].values:
            pass
        else:
            df_all_test_results = aggregate_results_across_folds(
                df_all_test_results, y_true, y_pred, post_ids, fold=k, reset_index=True
            )

        # Store optimal hyperparams
        all_optimal_hyperparams.append(optimal_hyperparams)

        # Store optimal models, for each k
        all_best_models.append(model)

    # Get mean scores
    train_score = train_scores.mean()
    val_score = val_scores.mean()

    # Evaluate all predictions on test set
    if (
        -1 in dataset["fold"].values
    ):  # Ensure it only runs on the test set, rather than fold
        y_true, y_pred, post_ids, = return_true_and_predicted_values_from_model_and_dataloader(
            model, test_loader, remove_padding=is_timeline_sensitive
        )
        df_all_test_results = aggregate_results_across_folds(
            df_all_test_results, y_true, y_pred, post_ids, fold=-1, reset_index=True
        )
    else:
        pass
    test_score = evaluate_df(df_all_test_results, metric=metric)

    scores = {}
    scores["train"] = train_score
    scores["val"] = val_score
    scores["test"] = test_score
    scores = pd.DataFrame(pd.Series(scores)).T  # Return scores as a DataFrame
    scores.rename(index={0: model_name})

    config["train_last"] = train_score
    config["val_last"] = val_score
    config["test_last"] = test_score

    classification_report = classification_report_for_single_method_using_y(
        df_all_test_results["y_true"],
        df_all_test_results["y_pred"],
        model_name=model_name,
        target_names=["S", "E", "O"],
        zero_division=0,
        metrics=["precision", "recall", "f1-score"],
    )

    return (
        scores,
        df_all_test_results,
        all_best_models,
        all_optimal_hyperparams,
        classification_report,
    )


def recreate_dataloaders_with_batch_size(
    data_loader_config,
    train_loader,
    val_loader,
    train_and_val_loader_combined,
    test_loader,
):
    """
    In order to avoid inefficiently continuously recreating the dataloaders,
    when iterating over different possible batch sizes - we only recreate
    the dataloaders in the case where the batch size has changed.
    """

    # Recreate the dataloaders, if the batch size has changed
    (
        train_loader,
        val_loader,
        train_and_val_loader_combined,
        test_loader,
    ) = data_handler.get_train_val_test_dataloaders(
        data_loader_config["dataset"],
        test_folds=[data_loader_config["k"]],
        train_val_test_sizes=data_loader_config["train_val_test_sizes"],
        features=data_loader_config["features"],
        target=data_loader_config["target"],
        batch_size=data_loader_config["batch_size"],
        shuffle=data_loader_config["shuffle"],
        num_workers=data_loader_config["num_workers"],
        is_time_aware=data_loader_config["is_time_aware"],
        which_dataset=data_loader_config["dataset_name"],
        assign_folds_to_nans=False,
    )

    return (
        train_loader,
        val_loader,
        train_and_val_loader_combined,
        test_loader,
    )


def grid_search_using_validation_set(
    progress_bar,
    hyperparams_to_search={"learning_rate": [0.0001, 0.001, 0.01], "epochs": [100]},
    loss_fn=nn.CrossEntropyLoss(),
    model_name="model_name",
    save_model=False,
    verbose=True,
    metric="macro_f1",
    is_timeline_sensitive=False,
    apply_standard_scaling=False,
    perform_early_stopping=True,
    early_stopping_criterion="loss",
    experiment_name="default_experiment_name",
    display_progress_bar=True,
    config={},
    retrain_optimal_hyperparams_on_train_and_val=False,  # Retrain on training and validation set
    data_loader_config={},
):
    """
    Returns the best model, and the optimal set of hyper-parameters, along with
    best train and validation score using the model with the best performance
    on the validation set.
    """

    loss_fn_name = config.get("loss_function_type", "cross_entropy")
    # gamma = config.get("gamma", 2.0)
    # beta_class_balanced = config.get("beta_class_balanced", -0.9999)

    # Remove hyper-parameters to search, if the model doesn't accept
    # the given hyper-parameter
    hyperparams_to_search = return_only_valid_hyper_parameters_for_model(
        model_name, hyperparams_to_search, loss_name=loss_fn_name
    )

    # Define all combinations of hyper-parameters to search
    hyper_param_combinations = identify_all_combinations_of_dictionary_of_lists(
        hyperparams_to_search
    )
    hyper_param_combinations = list(hyper_param_combinations)

    # Loop over each combination of hyper-parameters
    best_val_score = -9999999  # Initialize to extremely low

    # So that we always recreate the dataloaders at the start
    last_batch_size = -9999999
    train_loader = None
    val_loader = None
    train_and_val_loader_combined = None
    test_loader = None
    for i, hyper_params in enumerate(hyper_param_combinations):
        if verbose:
            print(
                "\t\t[h={}/{}] ({})\tHyper-param combinations searched.".format(
                    i, len(hyper_param_combinations), model_name
                )
            )

        data_loader_config["batch_size"] = hyper_params["batch_size"]
        data_loader_config["patience"] = hyper_params["patience"]
        
        # Get train/ val/ test from specified k
        (
            train_loader,
            val_loader,
            train_and_val_loader_combined,
            test_loader,
        ) = recreate_dataloaders_with_batch_size(
            data_loader_config,
            train_loader,
            val_loader,
            train_and_val_loader_combined,
            test_loader,
        )
        
        model = model_selector(
            model_name=model_name, hyper_params=hyper_params
        )  # Recreate the model

        # Create loss function (e.g. cross entropy / focal loss/ class balanced focal loss)
        loss_fn_train, loss_fn_validation = create_loss_function(
            train_loader,
            val_loader,
            which_loss=loss_fn_name,
            gamma=hyper_params.get("gamma"),  # For focal loss.
            beta=hyper_params.get("beta_cb"),  # For class balanced losses.
        )

        if verbose:
            print(hyper_params)

        # Adam, with current learning rate
        optimizer = optim.Adam(
            model.parameters(),
            lr=get_default_hyper_params(hyper_params, "learning_rate"),
        )

        # Train the current model
        training_loop(
            progress_bar=progress_bar,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=get_default_hyper_params(hyper_params, "epochs"),
            optimizer=optimizer,
            loss_fn_train=loss_fn_train,
            loss_fn_validation=loss_fn_validation,
            config=config,
            n_classes=3,
            verbose=verbose,
            remove_padding=is_timeline_sensitive,
            perform_early_stopping=perform_early_stopping,
            early_stopping_criterion=early_stopping_criterion,
            is_timeline_sensitive=is_timeline_sensitive,
            experiment_name=experiment_name,
            patience=data_loader_config["patience"]
            # hyper_params=hyper_params
        )

        # Scores on training and validation
        train_score = evaluate_model(
            model=model,
            test_loader=train_loader,
            metric=metric,
            verbose=verbose,
            remove_padding=is_timeline_sensitive,
            loss_fn=loss_fn_train,
        )

        val_score = evaluate_model(
            model=model,
            test_loader=val_loader,
            metric=metric,
            verbose=verbose,
            remove_padding=is_timeline_sensitive,
            loss_fn=loss_fn_validation,
        )
        
        if verbose:
            print("{} on:".format(metric))
            print("\tTraining Set:\t{:.4f}".format(train_score))
            print("\tValidation Set:\t{:.4f}".format(val_score))
            print("------------")

        # Choose best trained model with best hyper-parameters to be the final model
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model
            best_test_loader = (
                test_loader  # This contains the test set with the optimal batch size
            )
            optimal_hyperparams = hyper_params
            train_score_for_best_model = train_score
            val_score_for_best_model = val_score

        if display_progress_bar:

            progress_bar()

    if retrain_optimal_hyperparams_on_train_and_val:
        model = model_selector(
            model_name=model_name, hyper_params=hyper_params
        )  # Recreate the model

        if verbose:
            print("Retraining with optimal hyper-parameters:\n\t", hyper_params)

        # Adam, with current learning rate
        optimizer = optim.Adam(
            model.parameters(),
            lr=get_default_hyper_params(hyper_params, "learning_rate"),
        )

        # Train the current model
        training_loop(
            progress_bar=progress_bar,
            model=model,
            train_loader=train_and_val_loader_combined,
            val_loader=val_loader,
            n_epochs=get_default_hyper_params(hyper_params, "epochs"),
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_fn_val=loss_fn_val,
            n_classes=3,
            verbose=verbose,
            remove_padding=is_timeline_sensitive,
            perform_early_stopping=perform_early_stopping,
            early_stopping_criterion=early_stopping_criterion,
            is_timeline_sensitive=is_timeline_sensitive,
            experiment_name=experiment_name,
            # hyper_params=hyper_params
        )

        best_model = model
        best_test_loader = (
            test_loader  # This contains the test set with the optimal batch size
        )

    if verbose:
        print(
            "-----------\nFinished grid search. Optimal model had the following:\n train score: {:.4f}\tval score: {:.4f}".format(
                train_score_for_best_model, val_score_for_best_model
            )
        )
        print(
            "--- Optimal Hyper-parameters were as follows: ----\n", optimal_hyperparams
        )

    return (
        best_model,
        train_score_for_best_model,
        val_score_for_best_model,
        optimal_hyperparams,
        best_test_loader,
    )


def identify_all_combinations_of_dictionary_of_lists(dicts):
    """
    Returns a generator object, which can be iterated over, which takes an
    input dictionary, and returns all combinations of the keys and
    values. Useful for identifying all combinations of hyper-parameters to
    search.

    Example usage:

    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """

    combinations_dict = (
        dict(zip(dicts, x)) for x in itertools.product(*dicts.values())
    )

    return combinations_dict


# def get_default_hyper_params(dict_hyper_params, which="learning_rate"):
#     """
#     Returns the specified hyperparameters, and if None are returned,
#     then returns from a default list.
#     """
#     h = dict_hyper_params.get(which)

#     if h == None:
#         if which == "learning_rate":
#             h = 0.01
#         elif which == "epochs":
#             h = 1
#         elif which == "dropout":
#             h = 0  # Does not apply dropout, by default

#     return h

"""
Functions below are for error analysis - joining post id back to the test data
"""


def return_unrolled_y_true(
    y_true,
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
    is_not_padding = y_true != padding_value  # Mask: True if not padding
    is_not_padding = is_not_padding[:, :, 0]  # Remove embedding dimension
    is_not_padding = is_not_padding.view(-1)  # Unroll the timeline dimension

    # Unroll the batch dimension
    y_true = y_true.view(-1, n_classes)

    # Label encoding
    y_true = torch.argmax(y_true, dim=1)

    # Remove padding
    y_true = y_true[is_not_padding]

    return y_true


def return_true_and_predicted_values_from_dataloader(
    test_loader, remove_padding=False, padding_value=-123.0, n_classes=3
):
    """
    Returns the true values and the predicted values by an input model for a
    given data_loader. Ensures the model does not get trained.

    TODO: Note that when batch size > 1, these values are actually padded to
    size of the longest timeline in the batch! As they are an array. Must ensure
    that these padded values are removed before evaluation.
    """
    y_trues = []
    with torch.no_grad():
        for _, (inputs, y_true) in enumerate(test_loader):

            # Unroll Tensors to a scalar 1D array of label-encoded predictions: Output is (n_samples)
            y_true = return_unrolled_y_true(
                y_true,
                remove_padding=remove_padding,
                padding_value=padding_value,
                n_classes=n_classes,
                max_seq_length=124,
            )

            y_trues.append(y_true)

        actuals = torch.cat(y_trues)

    return actuals


def aggregate_results_across_folds(
    df_all_results, y_true, y_pred, post_ids, fold=0, reset_index=True
):
    """
    Stores results over different folds into a single DataFrame, which will
    then be used to evaluate the models across the whole dataset.
    """
    # Move to CPU, so place in Pandas Series
    y_true = y_true.clone().cpu()
    y_pred = y_pred.clone().cpu()
    post_ids = post_ids.clone().cpu()

    # Convert one-hot arrays to 1D series, if they are multi-dimensional
    if y_true.ndim > 1:
        y_true = data_handler.convert_one_hot_array_back_to_labels_df(
            y_true, reverse_one_hot=True
        )
    if y_pred.ndim > 1:
        y_pred = data_handler.convert_one_hot_array_back_to_labels_df(
            y_pred, reverse_one_hot=True
        )

    # Convert to Series, if not already:
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)
    if not isinstance(post_ids, pd.Series):
        post_ids = pd.Series(post_ids)

    # Create DataFrame for current set of results
    df_current_results = pd.DataFrame()
    df_current_results["y_true"] = y_true
    df_current_results["y_pred"] = y_pred
    df_current_results["fold"] = fold
    df_current_results["post_indexes"] = post_ids

    # Aggregate to full DataFrame, across all folds
    if len(df_all_results) == 0:
        df_all_results = df_current_results
    else:
        df_all_results = pd.concat([df_all_results, df_current_results], axis=0)

    # Reset the index
    if reset_index:
        df_all_results = df_all_results.reset_index().drop("index", axis=1)

    return df_all_results


def aggregate_results_across_folds_only_y_true(
    df_all_results, y_true, fold=0, reset_index=True
):
    """
    Stores results over different folds into a single DataFrame, which will
    then be used to evaluate the models across the whole dataset.
    """

    # Convert one-hot arrays to 1D series, if they are multi-dimensional
    if y_true.ndim > 1:
        y_true = data_handler.convert_one_hot_array_back_to_labels_df(
            y_true, reverse_one_hot=True
        )

    # Convert to Series, if not already:
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    # Create DataFrame for current set of results
    df_current_results = pd.DataFrame()
    df_current_results["y_true"] = y_true
    df_current_results["fold"] = fold

    # Aggregate to full DataFrame, across all folds
    if len(df_all_results) == 0:
        df_all_results = df_current_results
    else:
        df_all_results = pd.concat([df_all_results, df_current_results], axis=0)

    # Reset the index
    if reset_index:
        df_all_results = df_all_results.reset_index().drop("index", axis=1)

    return df_all_results
