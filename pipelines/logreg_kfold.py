import sys


from utils.kfold import kfold
from models import model_selector

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

from utils.io import data_handler


def logreg_experiment(
    which_data="both",
    prototype=True,
    hyperparams_to_search={"learning_rate": [0.01, 0.1], "epochs": [1, 10]},
    features=["sentence-bert"],
    target="label_3",
    model_name="logreg",
):
    datasets = get_datasets(which_data=which_data, prototype=prototype)

    # Train and evaluate on each dataset
    for dataset_name, dataset in datasets.items():
        model = model_selector(model_name=model_name)
        (
            scores,
            df_all_test_results,
            all_best_models,
            all_optimal_hyperparams,
            classification_report,
        ) = kfold(
            model=model,
            dataset=dataset,
            k_folds=5,
            features=features,
            target=target,
            train_val_test_sizes=[3, 1, 1],
            batch_size=1,
            shuffle=False,
            num_workers=1,
            hyperparams_to_search=hyperparams_to_search,
            metric="macro_f1",
            save_models=False,
            model_name=model_name,
        )

    return (
        scores,
        df_all_test_results,
        all_best_models,
        all_optimal_hyperparams,
        classification_report,
    )


def get_datasets(which_data="both", prototype=True, load=True):
    datasets = {}
    RedditDataset = data_handler.RedditDataset(
        include_embeddings=True,
        save_processed_timelines=False,
        load_timelines_from_saved=load,
    )
    TalkLifeDataset = data_handler.TalkLifeDataset(
        include_embeddings=True,
        save_processed_timelines=False,
        load_timelines_from_saved=load,
    )

    # Mini timelines
    if prototype:
        talklife_timelines = TalkLifeDataset.prototype_timelines
        reddit_timelines = RedditDataset.prototype_timelines
    else:
        talklife_timelines = TalkLifeDataset.timelines
        reddit_timelines = RedditDataset.timelines

    if which_data == "both":
        datasets["talklife"] = talklife_timelines
        datasets["reddit"] = reddit_timelines
    elif which_data == "talklife":
        datasets["talklife"] = talklife_timelines
    elif which_data == "reddit":
        datasets["reddit"] = reddit_timelines

    return datasets

