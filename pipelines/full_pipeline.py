import sys

import pandas as pd

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

from utils.io import data_handler
from utils.io.data_handler import (
    aggregate_dataframe_to_timeline_level,
    set_random_seeds,
)
from utils.io.my_pickler import my_pickler

sys.path.insert(0, "../../predicting_mocs/")
from models.model_selector import (
    check_if_model_is_time_aware,
    check_if_model_is_timeline_sensitive,
    model_selector,
)
from utils.evaluation import (
    concatenate_classification_reports_for_multiple_methods,
    sort_classification_reports,
)
from utils.kfold import kfold


def full_pipeline_for_single_model(
    progress_bar,
    model_name="model_name",
    which_data="talklife",
    prototype=False,
    hyperparams_to_search={"learning_rate": [0.01, 0.1], "epochs": [1, 10]},
    features=["sentence-bert"],
    target="label_3",
    random_seed=0,
    verbose=False,
    apply_standard_scaling=False,
    perform_early_stopping=True,
    early_stopping_criterion="loss",
    experiment_name="default_experiment_name",
    display_progress_bar=True,
    config={},
    folds='all'
):
    """
    Carries out full training, val, testing pipeline for a
    generic model, using just the string  name of the model
    which will then get called by the  model_selector to
    instantiate the class.
    """
    set_random_seeds(random_seed)  # Set random seed, for reproducibility
    is_timeline_sensitive = check_if_model_is_timeline_sensitive(model_name=model_name)
    is_time_aware = check_if_model_is_time_aware(model_name=model_name)

    if prototype:
        print(
            "WARNING: using smaller prototype dataset. Use full dataset, for more robust experiments."
        )
    if verbose:
        print("`{}`: Carrying out full experiment pipeline...".format(model_name))
        if is_timeline_sensitive:
            print(
                "`{}` is timeline sensitive. Aggregating dataset on timeline level".format(
                    model_name
                )
            )
        if is_time_aware:
            print(
                "`{}` is time-aware. Timestamps will be appended to dataloaders.".format(
                    model_name
                )
            )

    # Load all datasets (TalkLife, Reddit)
    datasets = get_datasets(
        which_data=which_data,
        prototype=prototype,
        aggregate_to_timeline_level=is_timeline_sensitive,
        apply_padding=True,
        max_seq_length=124,
        padding_value=-123.0,
        embedding_type=features[0],
    )

    # Train and evaluate on each dataset
    for dataset_name, dataset in datasets.items():
        # model = model_selector(model_name=model_name)
        (
            scores,
            df_all_test_results,
            all_best_models,
            all_optimal_hyperparams,
            classification_report,
        ) = kfold(
            progress_bar=progress_bar,
            model=None,
            dataset=dataset,
            dataset_name=dataset_name,
            features=features.copy(),
            target=target,
            train_val_test_sizes=[3, 1, 1],
            shuffle=False,
            num_workers=1,
            hyperparams_to_search=hyperparams_to_search,
            metric="macro_f1",
            save_models=False,
            model_name=model_name,
            is_timeline_sensitive=is_timeline_sensitive,
            is_time_aware=is_time_aware,
            verbose=verbose,
            apply_standard_scaling=apply_standard_scaling,
            perform_early_stopping=perform_early_stopping,
            early_stopping_criterion=early_stopping_criterion,
            experiment_name=experiment_name,
            display_progress_bar=True,
            config=config,
            folds=folds
        )

    return (
        scores,
        df_all_test_results,
        all_best_models,
        all_optimal_hyperparams,
        classification_report,
    )


def full_pipeline_for_multiple_models(
    progress_bar,
    model_names=["model_name"],
    which_data="talklife",
    prototype=False,
    hyperparams_to_search={"learning_rate": [0.01, 0.1], "epochs": [1, 10]},
    features=["sentence-bert"],
    target="label_3",
    verbose=False,
    apply_standard_scaling=False,
    perform_early_stopping=True,
    early_stopping_criterion="loss",
    experiment_name="default_experiment_name",
    display_progress_bar=True,
    config={},
    folds='all',
):
    """
    Train/val/test pipeline for multiple input models. Evalautes them, and
    places their results in an ordered classification report - sorted by
    macro-avg F1 score.
    """

    results = {}
    classification_reports = pd.DataFrame()
    for i, model_name in enumerate(model_names):
        if display_progress_bar:
            config["model_i"] = i
            config["model_name"] = model_name
            config["number_of_models"] = len(model_names)
        (
            scores,
            df_all_test_results,
            all_best_models,
            all_optimal_hyperparams,
            classification_report,
        ) = full_pipeline_for_single_model(
            progress_bar=progress_bar,
            model_name=model_name,
            which_data=which_data,
            prototype=prototype,
            hyperparams_to_search=hyperparams_to_search.copy(),
            features=features,
            target=target,
            verbose=verbose,
            random_seed=config["random_seed"],
            apply_standard_scaling=apply_standard_scaling,
            perform_early_stopping=perform_early_stopping,
            early_stopping_criterion=early_stopping_criterion,
            experiment_name=experiment_name,
            display_progress_bar=display_progress_bar,
            config=config,
            folds=folds
        )

        # Store outputs, for given model
        results[model_name] = {}
        results[model_name]["scores"] = scores
        results[model_name]["df_all_test_results"] = df_all_test_results
        # results[model_name]["all_best_models"] = all_best_models
        results[model_name]["all_optimal_hyperparams"] = all_optimal_hyperparams
        results[model_name]["classification_report"] = classification_report
        results[model_name]["config"] = config

        # Concatenate classification reports for multiple models
        classification_reports = (
            concatenate_classification_reports_for_multiple_methods(
                single_report=classification_report, all_reports=classification_reports
            )
        )

    # Sort classification reports
    classification_reports = sort_classification_reports(
        classification_reports, by=[("macro avg", "F1")], ascending=False
    )

    return classification_reports, results


def get_datasets(
    which_data="both",
    prototype=True,
    load=True,
    aggregate_to_timeline_level=False,
    apply_padding=True,
    max_seq_length=124,
    padding_value=-123.0,
    embedding_type="sentence-bert",
):
    datasets = {}
    if which_data == 'both':
        reddit_timelines = my_pickler(
            "i", "reddit_timelines_all_embeddings", folder="datasets"
        )
        # reddit_timelines = my_pickler(
        #     "i", "df_reddit_linguistic", folder="datasets"
        # )
        talklife_timelines = my_pickler(
            "i", "talklife_timelines_all_embeddings", folder="datasets"
        )
    elif which_data == 'reddit':
        reddit_timelines = my_pickler(
            "i", "reddit_timelines_all_embeddings", folder="datasets"
        )
        # reddit_timelines = my_pickler(
        #     "i", "df_reddit_linguistic", folder="datasets"
        # )
        
        # Assign integer post indexes (PyTorch expects integer values, not strings)
        reddit_timelines = reddit_timelines.reset_index().rename(columns={"index":'post_index'})
        
        # Convert train_or_test to -1 fold
        reddit_timelines.loc[reddit_timelines["train_or_test"] == "test", "fold"] = -1
        
        if aggregate_to_timeline_level:
            reddit_timelines = data_handler.aggregate_dataframe_to_timeline_level(
                reddit_timelines,
                features=[
                    embedding_type,
                    "time_epoch_days",
                    # "datetime",
                    # "label_5",
                    "label_3",
                    "label_2",
                    # "train_or_test",
                    # "user_id",
                    # "postid",
                    # "sentiment_vader","joy","optimism", "sadness","anger",  # linguistic features
                    # "emotion_embedding",
                    "post_index",
                    "fold",
                ],
                datatype="torch",
                apply_padding=apply_padding,
                max_seq_length=max_seq_length,
                padding_value=padding_value,
                embedding_type=embedding_type,
                which_dataset="reddit",
            )
            
        datasets["reddit"] = reddit_timelines


    elif which_data == 'talklife':
        talklife_timelines = my_pickler(
            "i", "talklife_timelines_all_embeddings", folder="datasets"
        )
        talklife_timelines = talklife_timelines.reset_index().rename(columns={"index":'post_index'})
        
        if aggregate_to_timeline_level:
            talklife_timelines = data_handler.aggregate_dataframe_to_timeline_level(
                talklife_timelines,
                features=[
                    embedding_type,
                    "time_epoch_days",
                    # "datetime",
                    "label_5",
                    "label_3",
                    "label_2",
                    # "user_id",
                    # "postid",
                    "post_index",
                    "fold",
                ],
                datatype="torch",
                apply_padding=apply_padding,
                max_seq_length=max_seq_length,
                padding_value=padding_value,
                embedding_type=embedding_type,
                which_dataset="talklife",
            )
        datasets["talklife"] = talklife_timelines

    else:
        # Load both datasets
        reddit_timelines = my_pickler(
            "i", "reddit_timelines_all_embeddings", folder="datasets"
        )
        # reddit_timelines = my_pickler(
        #     "i", "df_reddit_linguistic", folder="datasets"
        # )
        talklife_timelines = my_pickler(
            "i", "talklife_timelines_all_embeddings", folder="datasets"
        )

        # Assign integer post indexes (PyTorch expects integer values, not strings)
        reddit_timelines = reddit_timelines.reset_index().rename(columns={"index":'post_index'})
        talklife_timelines = talklife_timelines.reset_index().rename(columns={"index":'post_index'})

        # # Convert train_or_test to -1 fold
        reddit_timelines.loc[reddit_timelines["train_or_test"] == "test", "fold"] = -1

        if aggregate_to_timeline_level:
            talklife_timelines = data_handler.aggregate_dataframe_to_timeline_level(
                talklife_timelines,
                features=[
                    embedding_type,
                    "time_epoch_days",
                    # "datetime",
                    "label_5",
                    "label_3",
                    "label_2",
                    # "user_id",
                    # "postid",
                    "post_index",
                    "fold",
                ],
                datatype="torch",
                apply_padding=apply_padding,
                max_seq_length=max_seq_length,
                padding_value=padding_value,
                embedding_type=embedding_type,
                which_dataset="talklife",
            )
            reddit_timelines = data_handler.aggregate_dataframe_to_timeline_level(
                reddit_timelines,
                features=[
                    embedding_type,
                    "time_epoch_days",
                    # "datetime",
                    # "label_5",
                    "label_3",
                    "label_2",
                    # "sentiment_vader","joy","optimism", "sadness","anger",  # linguistic features
                    # "emotion_embedding",
                    # "train_or_test",
                    # "user_id",
                    # "postid",
                    "post_index",
                    "fold",
                ],
                datatype="torch",
                apply_padding=apply_padding,
                max_seq_length=max_seq_length,
                padding_value=padding_value,
                embedding_type=embedding_type,
                which_dataset="reddit",
            )
            
        datasets["talklife"] = talklife_timelines
        datasets["reddit"] = reddit_timelines
    

    return datasets
