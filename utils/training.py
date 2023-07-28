from decimal import Decimal

import numpy as np
import torch
from torch import nn, optim

# from data_handler import convert_labels_to_categorical

torch.set_default_dtype(torch.float64)

import sys

sys.path.insert(0, "../../predicting_mocs/")
from utils.evaluation import (
    evaluate_loss,
    evaluate_model,
    evaluate_predictions,
    return_unrolled_y_true_and_y_pred,
)
from utils.loss_functions import create_loss_function

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path
from utils.io.data_handler import load_model, save_model

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from global_parameters import device_to_use
device = torch.device(device_to_use if torch.cuda.is_available() else "cpu")



def training_loop(
    progress_bar,
    model,
    train_loader,
    n_epochs,
    optimizer,
    # hyper_params,
    config,
    loss_fn_train=nn.CrossEntropyLoss(),
    loss_fn_validation=nn.CrossEntropyLoss(),
    n_classes=3,
    verbose=True,
    remove_padding=False,
    padding_value=-123.0,
    perform_early_stopping=True,
    early_stopping_criterion="loss",
    val_loader=None,
    # metric_for_early_stopping="loss",
    is_timeline_sensitive=False,
    patience=5,
    experiment_name="default_experiment_name",
):
    """
    Trains the input model on the input train_loader.

    Standard training loop. Simply pass the desired number of epochs, train
    loader, chosen criterion (optimizer), and model you want to train, as well
    as the desired loss function.
    """

    # loss_fn = create_loss_function(which_loss=config["loss_function_type"])

    epochs_tolerated = 0
    best_score = 999999999999999999999999  # Initial high loss, to initialization
    best_epoch = 0  # Initial epoch, to initialization
    best_model_identified = False
    if verbose:
        print("--- Training for {} Epochs ---".format(n_epochs))
        # print("Results reported below are on the training set.")

    model.train()
    running_loss = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # Continue training with each epoch, until patience exceeded
        if not best_model_identified:
            (model, average_train_loss) = train_for_one_epoch(
                train_loader,
                optimizer,
                model,
                loss_fn_train,
                remove_padding=remove_padding,
                padding_value=padding_value,
                n_classes=n_classes,
                verbose=verbose,
            )

        # Perform early stopping, if val loss doesn't decrease
        if perform_early_stopping:
            best_model_identified = False
            # Check if patience exceeded. Continue training with each epoch, until patience exceeded.
            (
                best_model_identified,
                epochs_tolerated,
                best_score,
                best_epoch,
                val_score,
            ) = early_stopping(
                model=model,
                epochs_tolerated=epochs_tolerated,
                best_score=best_score,
                best_epoch=best_epoch,
                # metric=metric_for_early_stopping,
                val_loader=val_loader,
                epoch=epoch,
                is_timeline_sensitive=is_timeline_sensitive,
                patience=patience,
                model_name=experiment_name,
                loss_fn=loss_fn_validation,
                n_classes=3,
                early_stopping_criterion=early_stopping_criterion,
                config=config,
                # previous_epoch_train_loss=average_train_loss,
            )
            if best_model_identified:

                model = load_model(
                    model, experiment_name, file_type="torch"
                )  # Load the best model, and re-evaluate it again
                if verbose:
                    print(
                        "[epoch: {}] \t train loss: {:.2e} \t|\t val loss {:.2e}".format(
                            epoch + 1,
                            average_train_loss,
                            # average_train_f1,
                            val_score,
                        )
                    )

                    print(
                        "Loaded best model with early stopping, on validation set, found at epoch {}/{}".format(
                            best_epoch + 1, n_epochs
                        )
                    )
                    
                break

        # Print loss for current epoch
        if verbose:

            print(
                "[epoch: {}] \t train loss: {:.2e} \t|\t val loss {:.2e}".format(
                    epoch + 1,
                    average_train_loss,
                    # average_train_f1,
                    val_score,
                    # len(val_loader),
                )
            )


def train_for_one_epoch(
    train_loader,
    optimizer,
    model,
    loss_fn_train,
    remove_padding=True,
    padding_value=-123.0,
    n_classes=3,
    verbose=False,
):
    """
    If no early stopping, then train model and evaluate loss on
    training set, while performing backprop at the same time for
    each sample (batch) in the train_loader

    Args:
        train_loader (_type_): _description_
        optimizer (_type_): _description_
        model (_type_): _description_
        loss_fn_train (_type_): _description_
        remove_padding (bool, optional): _description_. Defaults to True.
        padding_value (float, optional): _description_. Defaults to -123.0.
        n_classes (int, optional): _description_. Defaults to 3.
        verbose (bool, optional): _description_. Defaults to False.
    """
    #
    loss_for_current_epoch = 0.0
    summed_f1_for_current_epoch = 0.0
    number_of_samples = 0
    for i, train_data in enumerate(train_loader, 0):
        inputs, y_true, post_index = train_data
        inputs, y_true, post_index = inputs.to(device), y_true.to(device), post_index.to(device)
        optimizer.zero_grad()

        # Forward
        y_pred = model(inputs)

        # Unroll Tensors to a scalars, while removing padding. Output is (n_samples) and (n_samples, n_classes)
        y_true, y_pred, _ = return_unrolled_y_true_and_y_pred(
            inputs,
            y_true,
            y_pred,
            post_index,
            remove_padding=remove_padding,
            padding_value=padding_value,
            n_classes=n_classes,
            max_seq_length=124,
            retain_predictions_as_probabilities=True,
        )

        # Backprop
        loss = loss_fn_train(y_pred, y_true)

        loss.backward()
        optimizer.step()  # Update weights

        # Store running loss, for simply printing statistics
        loss_for_current_epoch += loss.item()

        number_of_samples += y_pred.shape[
            0
        ]  # number of samples across all batches, y_pred is unpadded

    average_loss = loss_for_current_epoch / number_of_samples

    return model, average_loss


def early_stopping(
    model,
    epochs_tolerated,
    best_score,
    best_epoch,
    val_loader,
    is_timeline_sensitive,
    epoch,
    early_stopping_criterion="loss",
    patience=5,
    loss_fn=nn.CrossEntropyLoss(),
    n_classes=3,
    model_name="default_experiment_name",
    config={},
):
    """
    Takes an input trained model, and evaluates the validation loss. If the
    validation loss does not improve for `patience` epochs, then the model
    will stop training.

    Returns True if the best model has been identified early, from Patience.
    Otherwise, continuously saves the best model to a specified path.
    Later on outside the training loop, load from this path to load the best
    performing model.

    Patience was set to 5, in the original MoC paper, with 100 epochs.

    `epochs_tolerated` means number of epochs that we have been patient for.
    """
    best_model_identified = False
    # best_epoch = 0
    if model_name == "":
        model_name = "default_experiment_name"

    # Evaluate validation loss
    if early_stopping_criterion == "loss":
        val_score = evaluate_loss(
            model,
            val_loader,
            verbose=False,
            epochs=1,
            remove_padding=is_timeline_sensitive,
            loss_fn=loss_fn,
            n_classes=n_classes,
            return_as_average=True,
        )
    else:
        val_score = evaluate_model(
            model=model,
            test_loader=val_loader,
            metric=early_stopping_criterion,
            verbose=False,
            remove_padding=is_timeline_sensitive,
        )

    if val_score < best_score:  # Want to minimize loss
        best_score = val_score
        save_model(model, file_name=model_name, file_type="torch")
        epochs_tolerated = 0
        best_epoch = epoch
    else:
        epochs_tolerated += 1
        if epochs_tolerated >= patience:
            best_model_identified = True

    return (
        best_model_identified,
        epochs_tolerated,
        best_score,
        best_epoch,
        val_score,
    )  # , val_score
