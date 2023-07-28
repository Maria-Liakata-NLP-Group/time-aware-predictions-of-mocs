import datetime
import glob
import os
import sys
from lib2to3.pgen2.pgen import DFAState
from operator import truediv

# import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# import seaborn as sns


sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

from utils.io.my_pickler import my_pickler

sys.path.insert(0, "../../predicting_mocs/")
from utils.evaluation import (
    classification_report_for_single_method_using_y,
    sort_classification_reports,
)

sys.path.insert(0, "../../global_utils/")
import global_parameters
from global_parameters import path_default_pickle, path_figure_save_path


def return_all_results_as_dataframe(
    include_meta_data=True,
    include_clpsych_results=True,
    round_to_sig_fig=3,
    deduplicate=True,
    remove_reddit_new=True,
    save_results=False,
    only_top_two_clpsych=True,
    replace_model_names=True,
    include_batch_sizes=True,
    verbose=False,
    final_experiments_only=True
):
    """
    This is the main function you call to return all the stored results in a neatened form.
    """
    print("Loading all results as a dataframe...")
    
    df = return_all_stored_experiment_results_as_dataframe(
        include_meta_data=include_meta_data, verbose=verbose
    )
    df = df.copy()

    # Combine CLPsych
    if include_clpsych_results:
        df = pd.concat(
            [
                df,
                create_clpsych_df(
                    sort_by_macro_f1=True,
                    include_meta_data=True,
                    only_top_two_clpsych=only_top_two_clpsych,
                ),
            ],
            axis=0,
        )

    if round_to_sig_fig != None:
        df = df.round(int(round_to_sig_fig))

    # Get meta-data, e.g. loss function
    df["loss_function"] = df["source_file"].apply(get_loss_function_from_file_name)

    df["embedding_type"] = df["source_file"].apply(get_embedding_type_from_file_name)

    df["batch_sizes_searched"] = df["source_file"].apply(
        lambda x: get_batch_sizes_searched_from_file_name(x, default=[1], verbose=False)
    )

    # Subset (reddit-new)
    df = assign_data_subsets(df)
    if remove_reddit_new:
        df = df[df["subset"] != "reddit_new"]

    if deduplicate:
        df = deduplicate_results(df)

    # Reorder columns
    df = reorder_columns_in_results(df, reddit_new_removed=True)

    # Change model names
    if replace_model_names:
        df = replace_index_names(df)

    df = df.sort_values(by=("macro avg", "F1"), ascending=False)

    # Save results, if desired
    if save_results:
        df[df["which_dataset"] == "reddit"].to_csv("reddit_pmocs_results.txt", sep="\t")
        df[df["which_dataset"] == "talklife"].to_csv(
            "talklife_pmocs_results.txt", sep="\t"
        )
        df.to_csv("all_datasets_pmocs_results.txt", sep="\t")

    return df


def return_results_df_from_pickle(file_name, to_print=False, verbose=False):
    results = my_pickler(
        "i", file_name, folder="results", verbose=verbose, display_warnings=False
    )

    if to_print:
        print(results)

    return results


# def return_names_of_results():


def extract_file_name_from_full_path(full_path, is_in_results_directory=True):

    if is_in_results_directory:
        # split = full_path.split("../")
        split = full_path.split("/")
        file_name = split[-1]

    return file_name


def only_return_if_keyword_in_file_names(
    all_file_names, keyword="classification_report"
):
    output = []
    for file_name in all_file_names:
        if keyword in file_name:
            output.append(file_name)

    return output


def return_all_saved_results_file_names(
    only_classification_reports=True, which_dataset=None, only_auxiliary_results=False
):

    results_directory = path_default_pickle + "results/"
    all_file_names_in_results_full_path = glob.glob(results_directory + "*")

    file_names = []

    # Remove the prefix, and input into my_pickler
    for file_name in all_file_names_in_results_full_path:
        processed_file_name = extract_file_name_from_full_path(
            file_name, is_in_results_directory=True
        )
        file_names.append(processed_file_name)

    # Only return classification results file names
    if only_classification_reports:
        file_names = only_return_if_keyword_in_file_names(
            file_names, keyword="classification_report"
        )

    if only_auxiliary_results:
        file_names = only_return_if_keyword_in_file_names(
            file_names, keyword="auxiliary_results"
        )

    if which_dataset == None:
        pass
    else:
        file_names = only_return_if_keyword_in_file_names(file_names, which_dataset)


    return file_names


def concatenate_all_saved_results(
    include_creation_time=False,
    remove_duplicates=True,
    sort_by=[("macro avg", "F1")],
    which_dataset="reddit",
    remove_file_names_that_contain_substring=[],
    verbose=False,
    remove_specific_methods=[
        "lstm_heat_past_present_future",
        "lstm_heat_past_future_only",
        "logreg_differencing_heat_unnormalized_concat",
        "logreg_differencing_heat_normalized_concat",
    ],
    include_source_file=False,
):
    # Get all filenames
    file_names = return_all_saved_results_file_names(
        only_classification_reports=True, which_dataset=which_dataset
    )
    # print(len(file_names))

    # Filter any filenames that are not wanted, based on if they match a substring
    for f in file_names:
        if any(
            substring in f for substring in remove_file_names_that_contain_substring
        ):
            file_names.remove(f)

    # Extract all dataframes from file names
    list_all_dfs = []
    for f in file_names:
        # print(f)
        # if f == "import/nlp/ahills/pickles/results/classification_reports_reddit_2023_03_06_12_47_50____heat_bilstm_include_current_post_1_fl_bfl.pickle":
        #     pass
        # else:
        df = return_results_df_from_pickle(f, verbose=verbose, to_print=False)
        list_all_dfs.append(df)

        if include_source_file:
            df["source_file"] = f

    # Concatenate all the dataframes
    df = pd.concat(list_all_dfs)

    # Post-processing: remove duplicates
    if remove_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    # Remove specific methods, if desired
    for method in remove_specific_methods:
        df = df.drop(method, axis=0, errors="ignore")

    # Sort for desired metric
    df = sort_classification_reports(df, by=sort_by, ascending=False)

    if verbose:
        print(
            "Dataframes extracted for the following file names:\n{}".format(file_names)
        )

    return df


def plot_results(
    df,
    which_data="reddit",
    to_plot=("macro avg", "F1"),
    xlim=None,
    figsize=(10, 8),
    savefig=True,
    sort_results=True,
    file_format="pdf",
    methods_to_highlight=["logreg", "lstm_vanilla", "bilstm"],
    highlight_colour="red",
    title_prefix="",
    show_plot=True,
    show_title=True,
    show_legend=True,
    show_x_label=True,
):
    """
    Takes an input DataFrame, and plots a single set of results as a
    (horizontal) bar chart. Used for assessing predictive performance
    for prediciting Switches, Escalations.
    """

    if to_plot == None:
        # Plot everything together
        df.plot(
            kind="barh",
            y=[("macro avg", "F1"), ("S", "F1"), ("E", "F1"), ("O", "F1")],
            figsize=(10, 8),
            color=colours,
        ).invert_yaxis()
        title = "{} {}".format(which_data, "F1 macro avg SEO")
    else:
        if sort_results:
            series_to_plot = df[to_plot].sort_values(ascending=False)
        else:
            series_to_plot = df[to_plot]

        title = "{}{} {}".format(title_prefix, which_data, to_plot)

        plt.rcParams["figure.dpi"] = 400

        # print()
        # colour_dict = return_colour_dict_highlight_methods_in_plot(series_to_plot, methods_to_highlight=[series_to_plot['logreg'], series_to_plot['lstm_vanilla'], series_to_plot['bilstm']], highlight_colour=highlight_colour)
        # print(series_to_plot.index)

        colours = apply_legends_to_model_names(series_to_plot)

        if figsize != None:
            series_to_plot.plot(
                kind="barh", figsize=figsize, color=colours
            ).invert_yaxis()
        else:
            series_to_plot.plot(kind="barh", color=colours).invert_yaxis()

    prettify_axis_and_title(which_data, to_plot)

    if not show_title:
        plt.title("")
    if not show_x_label:
        plt.xlabel("")  # TODO: doesn't do anything currently
    if show_legend:
        show_custom_legend()

    rename_method_tick_labels(series_to_plot)

    if savefig:
        save_path = "home/ahills/LongNLP/predicting_mocs/figures/"
        file_format = file_format
        full_path = save_path + title + "." + file_format
        save_figure(
            title=title,
            dpi=500,
            file_format=file_format,
            path=save_path,
            facecolor="white",
            transparent=False,
        )
        print("Figure saved at: `{}`".format(full_path))

    if show_plot:
        plt.show()


def return_colour_dict_highlight_methods_in_plot(
    series, methods_to_highlight=[], highlight_colour="red"
):
    """
    Highlights the scores of a given method with a different colour.
    """
    colour_dict = {}
    for method in methods_to_highlight:
        colour_dict[method] = highlight_colour
    return colour_dict


def load_and_plot_results(
    which_dataset="reddit",
    metrics_to_visualize=[("macro avg", "F1"), ("S", "F1"), ("E", "F1"), ("O", "F1")],
    replace_model_names=True,
    savefig=True,
    return_as_df=True,
    remove_file_names_that_contain_substring=[],
    file_format="pdf",
    remove_specific_methods=[
        "lstm_heat_past_present_future",
        "lstm_heat_past_future_only",
        "logreg_differencing_heat_unnormalized_concat",
        "logreg_differencing_heat_normalized_concat",
    ],
    show_plot=True,
    show_title=True,
):
    """
    Loads and plots all the pickled results for predicting GTMoCs.
    """
    # Concatenate all the saved results into a single dataframe
    df = concatenate_all_saved_results(
        include_creation_time=False,
        remove_duplicates=True,
        sort_by=[("macro avg", "F1")],
        which_dataset=which_dataset,
        remove_file_names_that_contain_substring=remove_file_names_that_contain_substring,
        verbose=False,
        remove_specific_methods=remove_specific_methods,
    )

    # Replace model names
    if replace_model_names:
        df = replace_index_names(df)

    for metric in metrics_to_visualize:
        plot_results(
            df,
            to_plot=metric,
            savefig=savefig,
            file_format=file_format,
            which_data=which_dataset,
            show_plot=show_plot,
            show_title=show_title,
            show_legend=True,
        )

    if return_as_df:
        return df


def return_all_results_as_processed(
    df=None,
    load_data=True,
    which_dataset="reddit",
    round_to_dp=3,
    highlight=True,
    which_to_highlight="max",
    highlight_color="lightgreen",
    remove_duplicates=True,
    sort_by=[("macro avg", "F1")],
    remove_file_names_that_contain_substring=[],
    verbose=False,
    remove_specific_methods=[
        "lstm_heat_past_present_future",
        "lstm_heat_past_future_only",
        "logreg_differencing_heat_unnormalized_concat",
        "logreg_differencing_heat_normalized_concat",
    ],
):
    if load_data:
        df = concatenate_all_saved_results(
            include_creation_time=False,
            remove_duplicates=remove_duplicates,
            sort_by=sort_by,
            which_dataset=which_dataset,
            remove_file_names_that_contain_substring=remove_file_names_that_contain_substring,
            verbose=False,
            remove_specific_methods=remove_specific_methods,
        )

    # Round the results to decimal places, if desired
    if round_to_dp != None:
        df = df.round(round_to_dp)

    # Highlight
    if highlight:
        df = highlight_all_scores(df, which_to_highlight, highlight_color)
        df = df.format(precision=round_to_dp)  # Round

    return df


def apply_legends_to_model_names(df):

    classes = [
        "LSTM (time-aware)",
        "LogReg (time-aware)",
        "Differencing (time-aware)",
        "Baselines",
        "Baselines with trivial adjustments",
    ]

    colour_names = {
        classes[0]: "green",
        classes[1]: "#fd798f",
        classes[2]: "black",
        classes[3]: "#2976bb",
        classes[4]: "#751973",
    }

    method_names = list(df.index)
    colours = []
    for m in method_names:
        class_type = classify_method_name(m)
        colour = colour_names[class_type]
        colours.append(colour)

    return colours


def show_custom_legend():
    custom_legend_elements = [
        Patch(label="LSTM (time-aware)", facecolor="green"),
        Patch(label="LogReg (time-aware)", facecolor="#fd798f"),
        Patch(label="Differencing (time-aware)", facecolor="black"),
        Patch(label="Baselines", facecolor="#2976bb"),
        Patch(label="Baselines with trivial adjustments", facecolor="#751973"),
    ]
    plt.legend(handles=custom_legend_elements, bbox_to_anchor=(1.05, 1.0))


def classify_method_name(method_name):
    mapper = {
        "$LSTM_{<,o,>}^{*}$": "LSTM (time-aware)",
        "$LSTM_{<,o}$": "LSTM (time-aware)",
        "$LSTM_{<,o}^{*}$": "LSTM (time-aware)",
        "$LSTM_{o,>}^{*}$": "LSTM (time-aware)",
        "$LSTM_{<,o,>}$": "LSTM (time-aware)",
        "$LSTM_{o,>}$": "LSTM (time-aware)",
        "$LSTM_{>}$": "LSTM (time-aware)",
        "$LSTM_{<,>}$": "LSTM (time-aware)",
        "$LSTM_{<}$": "LSTM (time-aware)",
        "$LSTM_{<}^{*}$": "LSTM (time-aware)",
        "$LSTM_{<,>}^{*}$": "LSTM (time-aware)",
        "$LSTM_{>}^{*}$": "LSTM (time-aware)",
        # Baselines
        "$LSTM$": "Baselines",
        "$LogReg$": "Baselines",
        "$BiLSTM$": "Baselines",
        # Baselines (with trivial adjustments)
        "$LSTM_{o}$": "Baselines with trivial adjustments",
        "$BiLSTM_{o}$": "Baselines with trivial adjustments",
        # Logistic Regression
        "$LogReg_{<,o}$": "LogReg (time-aware)",
        "$LogReg_{<,o}^{*}$": "LogReg (time-aware)",
        "$LogReg_{o,>}^{*}$": "LogReg (time-aware)",
        "$LogReg_{<}$": "LogReg (time-aware)",
        "$LogReg_{<,o,>}$": "LogReg (time-aware)",
        "$LogReg_{<,o,>}^{*}$": "LogReg (time-aware)",
        "$LogReg_{<,>}^{*}$": "LogReg (time-aware)",
        "$LogReg_{<}^{*}$": "LogReg (time-aware)",
        r"$\nabla$": "Differencing (time-aware)",
        r"$\nabla_{n}$": "Differencing (time-aware)",
    }

    if method_name in mapper.keys():
        label = mapper[method_name]
    else:
        label = None

    return label


def prettify_axis_and_title(which_data, xlabel):
    if which_data == "talklife":
        plt.title("TalkLife")
    elif which_data == "reddit":
        plt.title("Reddit")

    clf = xlabel[0]
    print(xlabel)
    print(clf)
    if clf == "S":
        clf = "Switch"
    elif clf == "E":
        clf = "Escalation"
    elif clf == "O":
        clf = "No Change"
    elif clf == "macro avg":
        clf = "Macro Average"

    plt.xlabel(clf + " (F1 Score)")

    # Limits
    if which_data == "talklife":
        if xlabel == (("macro avg", "F1")):
            plt.xlim(left=0.3)
        if xlabel == (("S", "F1")):
            plt.xlim(left=0)
        if xlabel == (("E", "F1")):
            plt.xlim(left=0)
        if xlabel == (("O", "F1")):
            plt.xlim(left=0.85, right=0.92)
    elif which_data == "reddit":
        if xlabel == (("macro avg", "F1")):
            plt.xlim(left=0.3, right=0.6)
        if xlabel == (("S", "F1")):
            plt.xlim()
        if xlabel == (("E", "F1")):
            plt.xlim(left=0.12, right=0.67)
        if xlabel == (("O", "F1")):
            plt.xlim(left=0.65, right=0.88)


def rename_method_tick_labels(series):
    """
    Takes an input Series to be plotted, and renames the ordered method names
    so that they are less verbose. Also plots with the new tick labels.
    """
    original_labels = list(series.index)
    # y = list(series.values)
    y_pos = np.arange(len(original_labels))

    # Specify locations of baseliknes
    baselines = ["$LSTM$", "$LSTM_{o}$", "$BiLSTM$", "$BiLSTM_{o}$", "$LogReg$"]

    # Store baseline indices, to add them again lalter
    baseline_indices = {}
    for b in baselines:
        baseline_indices[b] = original_labels.index(b)

    # Change labels
    labels = list(
        map(lambda x: x.replace("LSTM_", "").replace("LogReg_", ""), original_labels)
    )

    # Re-add the baselines
    for b_label, b_index in baseline_indices.items():
        labels[b_index] = b_label

    plt.yticks(y_pos, labels)


def visualize_f1_scores_for_both_datasets_landscape_subplots(
    replace_model_names=True,
    score_on_y_axis=True,
    reddit_top=True,
    file_format="pdf",
    savefig=True,
    metrics_to_visualize=[("macro avg", "F1"), ("S", "F1"), ("E", "F1"), ("O", "F1")],
    remove_specific_methods=[
        "lstm_heat_past_present_future",
        "lstm_heat_past_future_only",
        "logreg_differencing_heat_unnormalized_concat",
        "logreg_differencing_heat_normalized_concat",
    ],
    remove_file_names_that_contain_substring=[],
    show_title=True,
):
    """
    Creates 8 subplots as a single landscape figure. Top row corresponds to
    results on Reddit, and 2nd row for results on TalkLife. The column order
    is for macro-avg F1, Switch, Escalation, No Change.

    If `score_on_y_axis=True`, then the F1 score will be on the shared y-axis,
    and method names on x-axis. Otherwise, the shared axes will be on the x-axis.
    """

    # Initialize figure to contain subplots
    plt.figure(figsize=(25, 15))
    plt.rcParams.update({"font.size": 11})

    # Plot for results in each dataset
    subplot_index = 0
    datasets = ["reddit", "talklife"]
    for which_dataset in datasets:

        # Concatenate all the saved results into a single dataframe
        df = concatenate_all_saved_results(
            include_creation_time=False,
            remove_duplicates=True,
            sort_by=[("macro avg", "F1")],
            which_dataset=which_dataset,
            remove_file_names_that_contain_substring=remove_file_names_that_contain_substring,
            verbose=False,
            remove_specific_methods=remove_specific_methods,
        )

        # Replace model names
        if replace_model_names:
            df = replace_index_names(df)

        # Specify location for subplot in figure
        nrows = len(datasets)
        ncols = len(metrics_to_visualize)

        # Create each subplot, in location
        for metric in metrics_to_visualize:

            subplot_index += 1
            plt.subplot(
                nrows, ncols, subplot_index
            )  # , sharex=False, sharey=False, frameon=True)

            if subplot_index > ncols:
                show_x_label = True
            else:
                show_x_label = False

            if subplot_index == ncols:
                show_legend = True
            else:
                show_legend = False

            plot_results(
                df,
                to_plot=metric,
                savefig=False,
                file_format=file_format,
                which_data=which_dataset,
                show_plot=False,
                show_title=show_title,
                show_legend=show_legend,
                figsize=None,
                show_x_label=show_x_label,
            )

    # plt.tight_layout()

    if savefig:
        save_path = "home/ahills/LongNLP/predicting_mocs/figures/"
        file_format = file_format
        title = "{} subplots for {}".format(str(nrows * ncols), str(datasets))
        full_path = save_path + title + "." + file_format
        save_figure(
            title=title,
            dpi=500,
            file_format=file_format,
            path=save_path,
            facecolor="white",
            transparent=False,
        )
        print("Figure saved at: `{}`".format(full_path))

    plt.show()

    # return df


def place_dollars_around_string(string):
    return "${}$".format(string)


def replace_index_names(df):
    dict_mapper = {
        "lstm_concat_exclude_present_in_heat_past_present_future": "$LSTM_{<,o,>}^{*}$",
        "lstm_concat_heat_past_present": "$LSTM_{<,o}$",
        "lstm_concat_exclude_present_in_heat_past_present": "$LSTM_{<,o}^{*}$",
        "lstm_concat_exclude_present_in_heat_present_future": "$LSTM_{o,>}^{*}$",
        "lstm_concat_heat_past_present_future": "$LSTM_{<,o,>}$",
        "lstm_concat_heat_present_future": "$LSTM_{o,>}$",
        "lstm_heat_future_present": "$LSTM_{>}$",
        "lstm_concat_heat_past_future": "$LSTM_{<,>}$",
        "lstm_heat_past_present": "$LSTM_{<}$",
        "lstm_heat_past_only": "$LSTM_{<}^{*}$",
        "lstm_concat_exclude_present_in_heat_past_future": "$LSTM_{<,>}^{*}$",
        "lstm_heat_future_only": "$LSTM_{>}^{*}$",
        "lstm_concat_exclude_present_in_heat_past_present_future_1_layer": "$LSTM_{<,o,>}^{*}$",
        # "lstm_concat_exclude_present_in_heat_past_present_future_1_layer": "$LSTM_{<,o,>}^{* (1)}$",
        # Baselines
        "lstm_vanilla": "$LSTM$",
        "lstm_vanilla_concat_present": "$LSTM_{o}$",
        "bilstm": "$BiLSTM$",
        "bilstm_concat_present": "$BiLSTM_{o}$",
        # Logistic Regression
        "logreg": "$LogReg$",
        "logreg_concat_heat_past_present": "$LogReg_{<,o}$",
        "logreg_concat_exclude_present_in_heat_past_present": "$LogReg_{<,o}^{*}$",
        "logreg_concat_exclude_present_in_heat_present_future": "$LogReg_{o,>}^{*}$",
        "logreg_heat_past_present": "$LogReg_{<}$",
        "logreg_concat_heat_past_present_future": "$LogReg_{<,o,>}$",
        "logreg_concat_exclude_present_in_heat_past_present_future": "$LogReg_{<,o,>}^{*}$",
        "logreg_concat_exclude_present_in_heat_past_future": "$LogReg_{<,>}^{*}$",
        "logreg_heat_past_only": "$LogReg_{<}^{*}$",
        "logreg_differencing_heat_unnormalized": r"$\nabla$",
        "logreg_differencing_heat_normalized": r"$\nabla_{n}$",
    }

    df = df.rename(index=dict_mapper)

    return df


def return_all_results_as_markdown(
    which_dataset="reddit",
    round_to_dp=3,
    remove_duplicates=True,
    replace_method_names=True,
    sort_by=[("macro avg", "F1")],
    remove_file_names_that_contain_substring=[],
    verbose=False,
    remove_specific_methods=[
        "lstm_heat_past_present_future",
        "lstm_heat_past_future_only",
        "logreg_differencing_heat_unnormalized_concat",
        "logreg_differencing_heat_normalized_concat",
    ],
):
    df = concatenate_all_saved_results(
        include_creation_time=False,
        remove_duplicates=True,
        sort_by=[("macro avg", "F1")],
        which_dataset=which_dataset,
        remove_file_names_that_contain_substring=remove_file_names_that_contain_substring,
        verbose=False,
        remove_specific_methods=remove_specific_methods,
    )

    # Replace method names
    if replace_method_names:
        df = replace_index_names(df)

    # Round the results to decimal places, if desired
    if round_to_dp != None:
        df = df.round(round_to_dp)
        string = df.to_markdown(floatfmt="#.{}f".format(str(round_to_dp)))  # Round
    else:
        string = df.to_markdown()

    # Get results rows only
    string = "\\".join(string.split("\n")[2:])

    # Replace characters
    string = string.replace("|\\|", "\\")  # Replace with new line characters
    string = string.replace("|", "&")  # Replace vertical lines with &
    string = string[2:-2]  # Remove starting & and ending &

    # Replace method names

    return string


def highlight_scores(s, which="max", color="seagreen"):
    if which == "max":
        is_max = s == s.max()
    elif which == "2nd_max":
        is_max = s == s.nlargest(2)[-1]
        # is_max =
    elif which == "3rd_max":
        is_max = s == s.nlargest(3)[-1]
    elif which == "min":
        is_max = s == s.min()
    elif which == "2nd_min":
        is_max = s == s.nsmallest(2)[1]

    return ["background: {}".format(color) if cell else "" for cell in is_max]


def highlight_all_scores(df, which="max", color="seagreen"):
    df = df.style.apply(highlight_scores, which=which, color=color)
    df = df.apply(highlight_scores, which="2nd_max", color="lightgreen")
    df = df.apply(highlight_scores, which="3rd_max", color="darkseagreen")
    # df = df.apply(highlight_scores, which='2nd_min', color='orange')
    # df = df.apply(highlight_scores, which='min', color='grey')

    return df


# def highlight_all_scores_as_single_call(s):
#     if mask = s == s.max():
#         is_max = s == s.max()
#     elif which == '2nd_max':
#     elif s == s.nlargest(2):
#     elif s == s.nsmallest(1)
#     elif which == '2nd_min':
#         is_max = s == s.nsmallest(2)

#     return ['background: {}'.format(color) if cell else '' for cell in is_max]


def load_and_plot_results_for_each_category_individually(
    which_dataset="reddit",
    metrics_to_visualize=[("macro avg", "F1"), ("S", "F1"), ("E", "F1"), ("O", "F1")],
    savefig=True,
    return_as_df=True,
    remove_file_names_that_contain_substring=[],
    file_format="pdf",
    remove_specific_methods=[
        "lstm_heat_past_present_future",
        "lstm_heat_past_future_only",
        "logreg_differencing_heat_unnormalized_concat",
        "logreg_differencing_heat_normalized_concat",
    ],
    remove_duplicates=True,
    sort_by=[("macro avg", "F1")],
):
    """
    Loads and plots all the pickled results for predicting GTMoCs.
    """

    dict_all_dfs = get_dict_dataframes(
        which_dataset=which_dataset,
        remove_file_names_that_contain_substring=remove_file_names_that_contain_substring,
        remove_specific_methods=remove_specific_methods,
        remove_duplicates=remove_duplicates,
        sort_by=sort_by,
    )

    for file_name, df in dict_all_dfs.items():
        for metric in metrics_to_visualize:
            plot_results(
                df,
                to_plot=metric,
                savefig=savefig,
                file_format=file_format,
                which_data=which_dataset,
                title_prefix="{}    ".format(file_name),
            )

    if return_as_df:
        return dict_all_dfs


def get_dict_dataframes(
    which_dataset="reddit",
    remove_file_names_that_contain_substring=[],
    remove_specific_methods=[
        "lstm_heat_past_present_future",
        "lstm_heat_past_future_only",
        "logreg_differencing_heat_unnormalized_concat",
        "logreg_differencing_heat_normalized_concat",
    ],
    remove_duplicates=True,
    sort_by=[("macro avg", "F1")],
    verbose=False,
):

    # Get all filenames
    file_names = return_all_saved_results_file_names(
        only_classification_reports=True, which_dataset=which_dataset
    )

    # Filter any filenames that are not wanted, based on if they match a substring
    for f in file_names:
        if any(
            substring in f for substring in remove_file_names_that_contain_substring
        ):
            file_names.remove(f)

    # Extract all dataframes from file names
    dict_all_dfs = {}
    for f in file_names:
        df = return_results_df_from_pickle(f, verbose=verbose, to_print=False)

        if remove_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        # Remove specific methods, if desired
        for method in remove_specific_methods:
            df = df.drop(method, axis=0, errors="ignore")

        # Sort for desired metric
        df = sort_classification_reports(df, by=sort_by, ascending=False)

        dict_all_dfs[f] = df

    return dict_all_dfs


def save_figure(
    title="",
    dpi=500,
    file_format="pdf",
    path="home/ahills/LongNLP/predicting_mocs/figures/",
    facecolor="white",
    transparent=False,
):
    """
    Saves the figure
    """
    path = "../" * 100 + path  # Ensure is root, and then go to path
    save_path = path + title + "." + file_format

    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, facecolor=facecolor, transparent=transparent)
    return None


def get_meta_data_from_file_name(file_name):
    """
    Returns the date the experiment started running, from the metadata in the file name.
    """
    meta_data = {}
    meta_data["which_dataset"] = get_which_dataset_from_file_name(file_name)
    meta_data["time_of_experiment"] = get_time_of_experiment_from_file_name(file_name)

    return meta_data


def get_time_of_experiment_from_file_name(file_name, verbose=False):
    try:
        f = file_name.split("_")

        # Get time of experiment
        date = f[3:9]
        date = "/".join(date)
        creation_time = datetime.datetime.strptime(date, "%Y/%m/%d/%H/%M/%S")
    except:
        creation_time = None

    return creation_time


def get_which_dataset_from_file_name(file_name, verbose=False):

    f = file_name.split("_")
    which_dataset = f[2]
    which_dataset = which_dataset.lower()

    if (which_dataset != "reddit") and (which_dataset != "talklife"):
        which_dataset = None

    return which_dataset


def get_which_loss_function_from_file_name(file_name):
    """
    Might need to use file name to access auxiliary results config to see loss function accurately.
    """
    f = "_".join(file_name.split("_")[2:])
    f = "auxiliary_results_" + f

    aux = my_pickler("i", f, folder="results")

    return aux


def concatenate_results_from_both_datasets(verbose=False):
    # Combine
    initialized = False
    for which_data in ["reddit", "talklife"]:
        all_results_single_dataset = concatenate_all_saved_results(
            include_creation_time=False,
            remove_duplicates=False,
            sort_by=[("macro avg", "F1")],
            which_dataset=which_data,
            remove_file_names_that_contain_substring=[],
            verbose=verbose,
            # remove_specific_methods=[],
            include_source_file=True,
        )
        if initialized:
            df = pd.concat([df, all_results_single_dataset], axis=0)
        else:
            df = all_results_single_dataset
            initialized = True

    return df


def return_all_stored_experiment_results_as_dataframe(
    include_meta_data=True, verbose=False
):
    df = concatenate_results_from_both_datasets(verbose=verbose)

    if include_meta_data:
        df["time_of_experiment"] = df["source_file"].apply(
            get_time_of_experiment_from_file_name, verbose=verbose
        )
        df["which_dataset"] = df["source_file"].apply(
            get_which_dataset_from_file_name, verbose=verbose
        )

        df["last_modified"] = df["source_file"].apply(
            get_modification_time_from_file_name, verbose=verbose
        )

        # Estimated length of experiment (how long it took to run) in hours.
        # Note this will be overestimate, if multiple models trained in same experiment
        df["length_of_experiment_hours"] = (
            df["last_modified"] - df["time_of_experiment"]
        ).apply(lambda x: round(x.total_seconds() / (60 * 60), 2))

    df = df.sort_values(by=("macro avg", "F1"), ascending=False)

    return df


def assign_data_subsets(df):
    df["subset"] = None
    df.loc[
        (df["last_modified"] < datetime.datetime(2022, 10, 27))
        & (df["which_dataset"] == "reddit"),
        "subset",
    ] = "reddit_new"
    df.loc[
        (df["last_modified"] >= datetime.datetime(2022, 10, 27))
        & (df["which_dataset"] == "reddit"),
        "subset",
    ] = "full"

    return df

def get_final_results_only(df, time_of_final_results=datetime.datetime(2023, 5, 11)):
    return df[df["last_modified"] >= time_of_final_results]
    
def get_seed_from_file_name(file_name):
    
    pass

def deduplicate_results(df, deduplicate_batch_sizes_searched=True):

    df = df.sort_values(by=("macro avg", "F1"), ascending=False)

    df = df.reset_index()
    df = df.rename(columns={"index": "model"})
    df["batch_sizes_searched"] = df["batch_sizes_searched"].apply(str)
    if deduplicate_batch_sizes_searched:
        df = df.drop_duplicates(
            subset=[
                ("model", ""),
                ("which_dataset", ""),
                ("loss_function", ""),
                ("subset", ""),
                ("embedding_type", ""),
                ("batch_sizes_searched", ""),
            ],
            keep="first",
        )
    else:
        df = df.drop_duplicates(
            subset=[
                ("model", ""),
                ("which_dataset", ""),
                ("loss_function", ""),
                ("subset", ""),
                ("embedding_type", ""),
                # ("batch_sizes_searched", ""),
            ],
            keep="first",
        )

    df = df.set_index("model")

    return df
    # df


def create_clpsych_df(
    sort_by_macro_f1=True, include_meta_data=True, only_top_two_clpsych=True
):

    # Initialize df with correct columns
    df = return_all_stored_experiment_results_as_dataframe(include_meta_data=False)
    clpsych_df = df.iloc[:1, :].copy()

    if only_top_two_clpsych:
        # UoS
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.689
        clpsych_df[("macro avg", "R")] = 0.625
        clpsych_df[("macro avg", "F1")] = 0.649

        clpsych_df[("S", "P")] = 0.490
        clpsych_df[("S", "R")] = 0.305
        clpsych_df[("S", "F1")] = 0.376

        clpsych_df[("E", "P")] = 0.697
        clpsych_df[("E", "R")] = 0.630
        clpsych_df[("E", "F1")] = 0.662

        clpsych_df[("O", "P")] = 0.881
        clpsych_df[("O", "R")] = 0.940
        clpsych_df[("O", "F1")] = 0.909

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "UoS"})
        clpsych_df_full = clpsych_df.copy()

        # WResearch
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.625
        clpsych_df[("macro avg", "R")] = 0.579
        clpsych_df[("macro avg", "F1")] = 0.598

        clpsych_df[("S", "P")] = 0.362
        clpsych_df[("S", "R")] = 0.256
        clpsych_df[("S", "F1")] = 0.300

        clpsych_df[("E", "P")] = 0.646
        clpsych_df[("E", "R")] = 0.553
        clpsych_df[("E", "F1")] = 0.596

        clpsych_df[("O", "P")] = 0.868
        clpsych_df[("O", "R")] = 0.929
        clpsych_df[("O", "F1")] = 0.897

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "WResearch"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

    else:
        # BLUE
        clpsych_df[("macro avg", "P")] = 0.505
        clpsych_df[("macro avg", "R")] = 0.495
        clpsych_df[("macro avg", "F1")] = 0.499

        clpsych_df[("S", "P")] = 0.175
        clpsych_df[("S", "R")] = 0.171
        clpsych_df[("S", "F1")] = 0.173

        clpsych_df[("E", "P")] = 0.484
        clpsych_df[("E", "R")] = 0.433
        clpsych_df[("E", "F1")] = 0.457

        clpsych_df[("O", "P")] = 0.855
        clpsych_df[("O", "R")] = 0.882
        clpsych_df[("O", "F1")] = 0.868

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "BLUE"})
        clpsych_df_full = clpsych_df.copy()

        # IIITH
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.520
        clpsych_df[("macro avg", "R")] = 0.600
        clpsych_df[("macro avg", "F1")] = 0.519

        clpsych_df[("S", "P")] = 0.206
        clpsych_df[("S", "R")] = 0.524
        clpsych_df[("S", "F1")] = 0.296

        clpsych_df[("E", "P")] = 0.402
        clpsych_df[("E", "R")] = 0.630
        clpsych_df[("E", "F1")] = 0.491

        clpsych_df[("O", "P")] = 0.954
        clpsych_df[("O", "R")] = 0.647
        clpsych_df[("O", "F1")] = 0.771

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "IIITH"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

        # LAMA
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.552
        clpsych_df[("macro avg", "R")] = 0.525
        clpsych_df[("macro avg", "F1")] = 0.524

        clpsych_df[("S", "P")] = 0.166
        clpsych_df[("S", "R")] = 0.354
        clpsych_df[("S", "F1")] = 0.226

        clpsych_df[("E", "P")] = 0.609
        clpsych_df[("E", "R")] = 0.389
        clpsych_df[("E", "F1")] = 0.475

        clpsych_df[("O", "P")] = 0.882
        clpsych_df[("O", "R")] = 0.861
        clpsych_df[("O", "F1")] = 0.871

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "LAMA"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

        # NLP-UNED
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.493
        clpsych_df[("macro avg", "R")] = 0.518
        clpsych_df[("macro avg", "F1")] = 0.501

        clpsych_df[("S", "P")] = 0.189
        clpsych_df[("S", "R")] = 0.293
        clpsych_df[("S", "F1")] = 0.230

        clpsych_df[("E", "P")] = 0.414
        clpsych_df[("E", "R")] = 0.471
        clpsych_df[("E", "F1")] = 0.440

        clpsych_df[("O", "P")] = 0.876
        clpsych_df[("O", "R")] = 0.791
        clpsych_df[("O", "F1")] = 0.832

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "NLP-UNED"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

        # UArizona
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.525
        clpsych_df[("macro avg", "R")] = 0.507
        clpsych_df[("macro avg", "F1")] = 0.510

        clpsych_df[("S", "P")] = 0.142
        clpsych_df[("S", "R")] = 0.220
        clpsych_df[("S", "F1")] = 0.172

        clpsych_df[("E", "P")] = 0.561
        clpsych_df[("E", "R")] = 0.423
        clpsych_df[("E", "F1")] = 0.482

        clpsych_df[("O", "P")] = 0.872
        clpsych_df[("O", "R")] = 0.879
        clpsych_df[("O", "F1")] = 0.876

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "UArizona"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

        # UoS
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.689
        clpsych_df[("macro avg", "R")] = 0.625
        clpsych_df[("macro avg", "F1")] = 0.649

        clpsych_df[("S", "P")] = 0.490
        clpsych_df[("S", "R")] = 0.305
        clpsych_df[("S", "F1")] = 0.376

        clpsych_df[("E", "P")] = 0.697
        clpsych_df[("E", "R")] = 0.630
        clpsych_df[("E", "F1")] = 0.662

        clpsych_df[("O", "P")] = 0.881
        clpsych_df[("O", "R")] = 0.940
        clpsych_df[("O", "F1")] = 0.909

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "UoS"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

        # uOttawa-AI
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.505
        clpsych_df[("macro avg", "R")] = 0.530
        clpsych_df[("macro avg", "F1")] = 0.512

        clpsych_df[("S", "P")] = 0.213
        clpsych_df[("S", "R")] = 0.244
        clpsych_df[("S", "F1")] = 0.227

        clpsych_df[("E", "P")] = 0.402
        clpsych_df[("E", "R")] = 0.553
        clpsych_df[("E", "F1")] = 0.466

        clpsych_df[("O", "P")] = 0.899
        clpsych_df[("O", "R")] = 0.793
        clpsych_df[("O", "F1")] = 0.842

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "uOttawa-AI"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

        # WResearch
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.625
        clpsych_df[("macro avg", "R")] = 0.579
        clpsych_df[("macro avg", "F1")] = 0.598

        clpsych_df[("S", "P")] = 0.362
        clpsych_df[("S", "R")] = 0.256
        clpsych_df[("S", "F1")] = 0.300

        clpsych_df[("E", "P")] = 0.646
        clpsych_df[("E", "R")] = 0.553
        clpsych_df[("E", "F1")] = 0.596

        clpsych_df[("O", "P")] = 0.868
        clpsych_df[("O", "R")] = 0.929
        clpsych_df[("O", "F1")] = 0.897

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "WResearch"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

        # WWBP-SQT-lite
        clpsych_df = df.iloc[:1, :].copy()
        clpsych_df[("macro avg", "P")] = 0.508
        clpsych_df[("macro avg", "R")] = 0.509
        clpsych_df[("macro avg", "F1")] = 0.508

        clpsych_df[("S", "P")] = 0.231
        clpsych_df[("S", "R")] = 0.220
        clpsych_df[("S", "F1")] = 0.225

        clpsych_df[("E", "P")] = 0.440
        clpsych_df[("E", "R")] = 0.462
        clpsych_df[("E", "F1")] = 0.451

        clpsych_df[("O", "P")] = 0.852
        clpsych_df[("O", "R")] = 0.845
        clpsych_df[("O", "F1")] = 0.848

        clpsych_df = clpsych_df[
            [
                ("macro avg", "P"),
                ("macro avg", "R"),
                ("macro avg", "F1"),
                ("S", "P"),
                ("S", "R"),
                ("S", "F1"),
                ("E", "P"),
                ("E", "R"),
                ("E", "F1"),
                ("O", "P"),
                ("O", "R"),
                ("O", "F1"),
            ]
        ]

        clpsych_df = clpsych_df.rename(index={clpsych_df.index[0]: "WWBP-SQT-lite"})
        clpsych_df_full = pd.concat([clpsych_df_full, clpsych_df], axis=0)

    if sort_by_macro_f1:
        clpsych_df_full = clpsych_df_full.sort_values(
            by=("macro avg", "F1"), ascending=False
        )

    if include_meta_data:
        clpsych_df_full["source_file"] = "clpsych_results"
        clpsych_df_full["which_dataset"] = "reddit"

    return clpsych_df_full


def get_modification_time_from_full_path(path, verbose=False):
    """
    Gets the time when the file was last modified (i.e. created). No easy way
    to get when file was created on Linux, so this is the next best option.
    """

    t = os.path.getmtime(path)

    return datetime.datetime.fromtimestamp(t).replace(microsecond=0)


def get_modification_time_from_file_name(file_name, verbose=False):
    path = global_parameters.path_default_pickle + "results/"
    path += file_name
    path += ".pickle"
    
    # Remove path names with two pickles.
    if path[-14:] == ".pickle.pickle":
        path = path[:-7]

    t = get_modification_time_from_full_path(path)

    return t


def get_loss_from_file_name_explicitly(file_name, verbose=False):

    if "class_balanced_focal_loss" in file_name:
        loss_function = "class_balanced_focal_loss"
    elif ("focal_loss" in file_name) or ("fl" in file_name):
        loss_function = "focal_loss"
    else:
        loss_function = "cross_entropy"

    return loss_function


def get_loss_function_from_file_name_by_loading_aux(
    file_name, verbose=False, loss_or_embedding="loss"
):
    """
    Given an input file name (classification_reports), will return the loss function
    used for this experiment.
    """

    loss_function = None
    embedding_type = None
    try:
        # Convert file name, to the auxiliary stored meta-data file name
        aux = get_config_files_from_file_name(file_name, verbose=False)
        models = list(aux.keys())
        i = 0  # All models should share the same loss, for a given experiment
        model_name = models[i]
        meta_data_keys = list(aux[models[i]].keys())
        config = aux[model_name]

        if config != None:
            embedding_type = config.get("embedding_type")
            loss_function = config.get("loss_function_type")

    except:
        pass

    if loss_or_embedding == "loss":
        return loss_function
    else:
        return embedding_type


def get_config_files_from_file_name(file_name, verbose=False):
    """
    Takes an input file name and returns the config file used to train the model.
    """

    # Load the auxiliary results file
    f = "_".join(file_name.split("_")[2:])
    f = "auxiliary_results_" + f
    aux = my_pickler("i", f, folder="results", verbose=verbose)

    # Get config files for each model
    models = list(aux.keys())
    all_config_files = {}
    for m in models:
        # meta_data_keys = list(aux[m].keys())
        config = aux[m].get("config")
        all_config_files[m] = config

    return all_config_files


def get_loss_function_from_file_name(file_name, default_loss="n/a"):

    loss_function = None

    # Try loading the loss function from the stored meta-data of the experiment
    loss_function = get_loss_function_from_file_name_by_loading_aux(file_name)

    # If no stored loss function, then check if it was stored in the file name
    if loss_function == None:
        loss_function = get_loss_from_file_name_explicitly(file_name)

    # If still no loss is available to be identified, provide the default loss
    # function name
    if loss_function == None:
        loss_function = default_loss

    return loss_function


def get_embedding_type_from_file_name_by_loading_aux(file_name):
    loss_function = None
    try:
        aux = my_pickler("i", file_name, folder="results")
        models = list(aux.keys())
        model_name = models[i]
        meta_data_keys = list(aux[models[i]].keys())
        config = aux[model_name].get("config")

        if config != None:
            embedding_type = config.get("embedding_type")
            loss_function = config.get("loss_function_type")
    except:
        pass

    return embedding_type


def get_embedding_type_from_file_name_explicitly(file_name):
    if "bert_class_balanced_focal_loss" in file_name:
        embedding_type = "bert_class_balanced_focal_loss"
    elif ("bert_focal_loss" in file_name) or ("bfl" in file_name):
        embedding_type = "bert_focal_loss"
    else:
        embedding_type = "sentence_bert"

    return embedding_type


def get_embedding_type_from_file_name(file_name, default_embedding="n/a"):
    embedding_type = None

    embedding_type = get_loss_function_from_file_name_by_loading_aux(
        file_name, loss_or_embedding="embedding"
    )

    if embedding_type == None:
        embedding_type = get_embedding_type_from_file_name_explicitly(file_name)

    if embedding_type == None:
        embedding_type = default_embedding
    return embedding_type


def reorder_columns_in_results(df, reddit_new_removed=True):
    """
    Re-orders the columns in the reported dataframe of results.
    """
    all_cols = df.columns.tolist()
    cols = all_cols[:12]  # Basic metrics (macro f1, etc.)
    cols += [
        ("loss_function", ""),
        ("embedding_type", ""),
        ("batch_sizes_searched", ""),
        ("which_dataset", ""),
        ("time_of_experiment", ""),
        ("last_modified", ""),
        ("length_of_experiment_hours", ""),
        ("source_file", ""),
    ]

    if reddit_new_removed == False:
        cols += [("subset", "")]

    df = df[cols]

    return df


def get_batch_sizes_searched_from_file_name(file_name, default=[1], verbose=False):
    """
    Takes an input file name and returns the batch sizes searched over.

    Args:
        file_name (str): The file name of the model.
        default (list): The default batch sizes to return if the batch sizes cannot be found.

    Output:
        batch_sizes (list): The batch sizes searched over.
    """

    try:
        config_files = get_config_files_from_file_name(file_name, verbose=verbose)
        model_name = list(config_files.keys())[
            0
        ]  # slight hack. all models should share same config file currently
        batch_sizes = config_files[model_name]["hyper_params_to_search"]["batch_size"]
    except:
        batch_sizes = default

    return batch_sizes


def aggregate_and_evaluate_results_from_multiple_experiments(file_names, to_markdown=True):
    """
        file_names = ['auxiliary_results_reddit_2022_10_18_18_24_19____lstm_and_bilstm',
    'auxiliary_results_reddit_2022_10_19_09_18_08____lstm_and_bilstm_optimize_f1_not_loss',
    'auxiliary_results_reddit_2022_10_19_16_11_57____lstm_and_bilstm_prototype_dataset_progress_bar'
    ]
    """

    all_results = pd.DataFrame()

    # Aggregate results from different saved experiments
    for f in file_names:

        ar = my_pickler("i", f, folder="results")
        model_name = list(ar.keys())[0]
        print(ar[model_name].keys())
        results_for_current_fold = ar[model_name]["df_all_test_results"]

        # Combine predictions and true values across different folds
        all_results = pd.concat([all_results, results_for_current_fold], axis=0)
    all_results = all_results.reset_index().drop("index", axis=1)

    # Evaluate combined values
    classification_report = classification_report_for_single_method_using_y(
        all_results["y_true"],
        all_results["y_pred"],
        model_name=model_name,
        target_names=["S", "E", "O"],
        zero_division=0,
        metrics=["precision", "recall", "f1-score"],
    )
    
    if to_markdown:
        return classification_report.to_markdown()
    else:
        return classification_report


def return_seed_number_from_classification_report(df_classification_report):
    df = df_classification_report.copy()
    
    
    
    return df

def aggregate_and_evaluate_results_from_multiple_folds_and_seeds(file_names):
    """
        file_names = ['auxiliary_results_reddit_2022_10_18_18_24_19____lstm_and_bilstm',
    'auxiliary_results_reddit_2022_10_19_09_18_08____lstm_and_bilstm_optimize_f1_not_loss',
    'auxiliary_results_reddit_2022_10_19_16_11_57____lstm_and_bilstm_prototype_dataset_progress_bar'
    ]
    """

    all_results = pd.DataFrame()

    # Aggregate results from different saved experiments
    for f in file_names:

        ar = my_pickler("i", f, folder="results")
        model_name = list(ar.keys())[0]
        results_for_current_fold = ar[model_name]["df_all_test_results"]

        # Combine predictions and true values across different folds
        all_results = pd.concat([all_results, results_for_current_fold], axis=0)
    all_results = all_results.reset_index().drop("index", axis=1)

    # Evaluate combined values
    classification_report = classification_report_for_single_method_using_y(
        all_results["y_true"],
        all_results["y_pred"],
        model_name=model_name,
        target_names=["S", "E", "O"],
        zero_division=0,
        metrics=["precision", "recall", "f1-score"],
    )

    return classification_report.to_markdown()

def get_auxiliary_results_from_classification_report_file_name(file_name):

    # Load auxiliary results file, from classification results file name
    file_name_no_prefix = file_name[len('classification_reports'):]
    file_name_ar = 'auxiliary_results' + file_name_no_prefix
    
    return file_name_ar

def load_auxiliary_results_from_classification_report_file_name(file_name):
    file_name_ar = get_auxiliary_results_from_classification_report_file_name(file_name)
    ar = my_pickler("i", file_name_ar, folder="results")

    
    return ar

def get_seed_from_classification_report_file_name(file_name):
    
    ar = load_auxiliary_results_from_classification_report_file_name(file_name)
    
    model_name = list(ar.keys())[0]
    seed = ar[model_name]['config'].get('random_seed')
    
    return seed

def get_fold_number_from_file_name(file_name):
    ar = load_auxiliary_results_from_classification_report_file_name(file_name)
    
    model_name = list(ar.keys())[0]
    folds = ar[model_name]['config']['folds']
    
    return folds

def get_recent_renamed_results(full_df, dt=datetime.datetime(2023, 5, 12)):
    df = full_df.copy()

    df = df[df["time_of_experiment"] >= dt]

    # Rename
    df.index = df.index.map(return_clean_model_name_final_results)
    
    return df

# def join_predictions_with_stored_metadata(model_name='2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h', which_dataset='reddit', custom_ground_truth_pickle=''):
#     """
#     Joins the metadata (time-intervals) to the stored predictions of a model, 
#     for subsequent analysis.
#     """
    
    
#     file_name = 'error_analysis_df_all_test_results_' + model_name
    
#     # Load ground-truth data
#     predictions = my_pickler("i", file_name, folder='results')
    
    
    
#     # Load ground-truth data
#     if which_dataset == 'reddit':
#         if custom_ground_truth_pickle == '':
#             gt = my_pickler("i", "reddit_timelines_all_embeddings", folder='datasets')
#         else:
#             gt = my_pickler("i", custom_ground_truth_pickle, folder='datasets')
        
    
#     # Get metadata from test dataset
#     test_df = gt[gt['train_or_test'] == 'test']
#     metadata = test_df.reset_index().rename(columns={'index':'post_indexes'})
    
#     # Merge data
#     df = pd.merge(predictions, metadata, on='post_indexes', how='outer')
    
#     df = get_time_deltas(df)
    
#     return df

def join_predictions_with_metadata(df, model_name='HEAT', random_seed=0, which_dataset='reddit', custom_ground_truth_pickle=''):
    """
    Joins the metadata (time-intervals) to the stored predictions of a model, 
    for subsequent analysis.
    """
    
    # Filter to current model, random seed, and dataset
    df = df[(df['model_name'] == model_name) 
            & 
            (df['random_seed'] == random_seed)
            &
            (df['which_dataset'] == which_dataset)].copy()
    
    # Get the file names for the stored predictions, for the current model and random seed
    ar_file_names = list(df['auxiliary_file_name'].values)
    
    # Load predictions, for all seeds and folds
    initialized = False
    df_all_test_results_combined_for_all_folds = pd.DataFrame()
    for ar_fn in ar_file_names:
        # Load stored predictions
        ar = my_pickler("i", ar_fn, folder='results')
        model = list(ar.keys())[0]
        df_all_test_results = ar[model]['df_all_test_results']
        
        # Get random seed that was used
        df_all_test_results['random_seed'] = ar[model]['config']['random_seed']
        
        # Concatenate all test predictions, across different folds and seeds
        if initialized:
            df_all_test_results_combined_for_all_folds = pd.concat([df_all_test_results_combined_for_all_folds, 
                                                                    df_all_test_results], 
                                                                axis=0)
        else:
            df_all_test_results_combined_for_all_folds = df_all_test_results
            initialized = True    
    predictions = df_all_test_results_combined_for_all_folds
    
    # Load ground-truth data
    if which_dataset == 'reddit':
        if custom_ground_truth_pickle == '':
            gt = my_pickler("i", "reddit_timelines_all_embeddings", folder='datasets')
        else:
            gt = my_pickler("i", custom_ground_truth_pickle, folder='datasets')
    elif which_dataset == 'talklife':
        if custom_ground_truth_pickle == '':
            gt = my_pickler("i", "talklife_timelines_all_embeddings", folder='datasets')
        else:
            gt = my_pickler("i", custom_ground_truth_pickle, folder='datasets')
        
    
    # Get metadata from test dataset
    # test_df = gt[gt['train_or_test'] == 'test']
    metadata = gt.reset_index().rename(columns={'index':'post_indexes'})
    
    # Merge data
    df = pd.merge(predictions, metadata, on='post_indexes', how='outer')
    
    # Remove rows with no predictions
    df = df[df['y_pred'].notna()]
    
    df = get_time_deltas(df)
    
    return df


def get_time_deltas(df):
    # Get time-intervals to previous post
    df['time_delta'] = df.sort_values(['timeline_id','time_epoch_days']).groupby('timeline_id')['time_epoch_days'].diff().fillna(0)
    
    return df

def scores_per_time_interval(df, time_intervals_to_search=list(range(30)), model_name='2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h'):
                                
    all_df = pd.DataFrame()
    df = df.copy()
    df = df[df['model_name'] == model_name]

    for tau in time_intervals_to_search:

        # Filter posts with less than or equal specified time-interval
        df_less_than_tau = df[df['time_delta'] <= tau]


        # Evaluate posts
        classification_report = classification_report_for_single_method_using_y(
                df_less_than_tau["y_true"],
                df_less_than_tau["y_pred"],
                model_name=model_name,
                target_names=["S", "E", "O"],
                zero_division=0,
                metrics=["precision", "recall", "f1-score"],
            )

        # Add tau to classification report
        classification_report['tau'] = tau

        all_df = pd.concat([all_df, classification_report], axis=0)
                                
    # Get model_name
    all_df = all_df.reset_index().rename(columns={'index':'model_name'})
                                    
                                
    return all_df




def plot_filtered_df_time_intervals(df, x_max=6, clean_name='-BDR'):
    styles = ['b-.', 'b--', 'b-',  # macro avg 
             'r-.', 'r--', 'r-',   # switch
             'g-.', 'g--', 'g-',   # escalation
             'k-.', 'k--', 'k-']  # no change

    df_to_plot = df.drop('file_name', axis=1)
    df_to_plot = df[df['tau'] <= x_max]
    df_to_plot.plot(x='tau', style=styles)
    plt.legend(loc=(1.04, 0))
    plt.title(clean_name)
    plt.ylabel('Score')
    plt.xlabel('Time-interval (days)')
    plt.show()
    
    
def load_and_plot(model_name='2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h', dataset='reddit', time_intervals_to_search=list(range(30)), x_max=6):
    df = join_predictions_with_metadata(model_name=model_name, which_dataset=dataset)
    df_scores = scores_per_time_interval(df, time_intervals_to_search=time_intervals_to_search)
    
    # Plot
    plot_filtered_df_time_intervals(df_scores, x_max=x_max, clean_name=get_clean_name_from_model_name(model_name))
    
    
def get_clean_name_from_model_name(model_name):
    if 'tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy' in model_name:
        clean = 'HEAT'
    elif '1_layer_bilstm_cross_entropy' in model_name:
        clean = 'BiLSTM'
    elif 'heat_no_bdr_concat_h' in model_name:
        clean = '-BDR'
    elif 'heat_no_msd' in model_name:
        clean = '-MSD'
        
    return clean

def return_clean_model_name_final_results(model_name):
    if model_name == 'bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh':
        clean = 'HEAT'
    elif model_name == 'bilstm_single_layer':
        clean = 'BiLSTM'
    elif model_name == 'heat_lstm_no_bdr_concat_h':
        clean = '-BDR'
    elif model_name == 'heat_no_msd':
        clean = '-MSD'
    else:
        clean = model_name
        
    return clean

# def return_dataframe_with_filtered_time_intervals(df_meta_data, models=["2023_03_28_17_00_08____tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy",
#            "2023_04_05_16_14_43____1_layer_bilstm_cross_entropy",
#            "2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h",
#            "2023_04_15_15_35_02____reddit_heat_no_msd"], which_dataset='reddit', time_intervals_to_search=list(range(30)), x_max=30):
    
#     all_df = pd.DataFrame()
#     for mn in models:
        
#         df = join_predictions_with_metadata(df_meta_data, model_name=mn, which_dataset=which_dataset)
#         df_scores = scores_per_time_interval(df, time_intervals_to_search=time_intervals_to_search, model_name=mn)
        
#         all_df = pd.concat([all_df, df_scores], axis=0)
        
#     return all_df

def return_dataframe_with_filtered_time_intervals(df_meta_data, time_intervals_to_search=list(range(30)), x_max=30):
    
    all_df = pd.DataFrame()
    models = df_meta_data['model_name'].unique().tolist()
    for mn in models:        
        df_scores = scores_per_time_interval(df_meta_data, time_intervals_to_search=time_intervals_to_search, model_name=mn)
        all_df = pd.concat([all_df, df_scores], axis=0)
        
    return all_df

def plot_horizontal_multiple_metrics_per_model(which_dataset='reddit', max_time_interval_days=5, metric=('macro avg', 'F1')):
        
    t_intervals = list(np.array(list(range(max_time_interval_days*10+1))) * 0.1)
    x_max = t_intervals[-1]
    df = return_dataframe_with_filtered_time_intervals(models=["2023_03_28_17_00_08____tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy",
           "2023_04_05_16_14_43____1_layer_bilstm_cross_entropy",
           "2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h",
           "2023_04_15_15_35_02____reddit_heat_no_msd"], which_dataset=which_dataset, time_intervals_to_search=t_intervals, x_max=x_max)
    
    all_metrics = [('macro avg', 'F1'), ('S', 'F1'), ('E', 'F1'), ('O', 'F1')]

    grouped = df.groupby('model_name')
    rowlength = 2                       # fix up if odd number of groups
    fig, axs = plt.subplots(figsize=(8,8), 
                            nrows=2, ncols=rowlength,     # fix as above
                            gridspec_kw=dict(hspace=0.4), # Much control of gridspec
                            sharex=True, sharey=True) 

    targets = zip(grouped.groups.keys(), axs.flatten())
    for i, (key, ax) in enumerate(targets):
        for metric in all_metrics:
            ax.plot(grouped.get_group(key)['tau'], grouped.get_group(key)[metric], label=metric)
            ax.set_title(key)
    ax.legend()
    plt.show()
    
    
def plot_multiple_models_one_plot_one_metric(metric = ('macro avg', 'F1'), 
                                             which_dataset = 'reddit', 
                                             max_time_interval_days = 5, 
                                             model_names = ["2023_03_28_17_00_08____tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy",
           "2023_04_05_16_14_43____1_layer_bilstm_cross_entropy", 
           "2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h",
           "2023_04_15_15_35_02____reddit_heat_no_msd"]):
    
    t_intervals = list(np.array(list(range(max_time_interval_days*10+1))) * 0.1)
    x_max = t_intervals[-1]

    df_tau = return_dataframe_with_filtered_time_intervals(models=model_names, which_dataset=which_dataset, time_intervals_to_search=t_intervals, x_max=x_max)
    
    fig, ax = plt.subplots()

    for key, grp in df_tau.groupby(['model_name']):
        ax = grp.plot(ax=ax, kind='line', x='tau', y=metric, label=key, ylabel=metric, xlabel='Time Interval (days)', title=which_dataset)
        
    plt.show() 
    

# def subplot_3_by_3(models=["2023_03_28_17_00_08____tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy",
#                            "2023_04_05_16_14_43____1_layer_bilstm_cross_entropy",
#                            "2023_04_15_16_39_50____reddit_heat_no_bdr_concat_h",
#            "2023_04_15_15_35_02____reddit_heat_no_msd"], figsize=(15,15), shared_axes=True, which_dataset='reddit', max_time_interval_days=5):
    
#     t_intervals = list(np.array(list(range(max_time_interval_days*10+1))) * 0.1)
#     x_max = t_intervals[-1]
#     df = return_dataframe_with_filtered_time_intervals(models=models, which_dataset=which_dataset, time_intervals_to_search=t_intervals, x_max=x_max)
    
#     # Define hyper-parameters
#     all_model_names = ['HEAT', 'BiLSTM', '-BDR', '-MSD']
#     all_metrics = [('macro avg', 'F1'), ('S', 'F1'), ('E', 'F1'), 
#                 ('macro avg', 'P'), ('S', 'P'), ('E', 'P'), 
#                 ('macro avg', 'R'), ('S', 'R'), ('E', 'R')]
    
#     row_length=3
#     fig, ax = plt.subplots(figsize=figsize, 
#                         nrows=row_length, ncols=row_length,     # fix as above
#                         # gridspec_kw=dict(hspace=0.4), # Much control of gridspec
#                         sharex=shared_axes, sharey=shared_axes)

#     for m, row in zip(['F1', 'P', 'R'], ax):
#         for c, col in zip(['macro avg', 'S', 'E'], row):
#             for model_name in all_model_names:
                
#                 metric = (c, m)
            
#                 y = df[df['model_name'] == model_name][metric]
#                 x = df[df['model_name'] == model_name]['tau']

#                 col.plot(x, y, label=model_name)
                
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.legend(loc=(-1.5, 2.25))

#     # Column Headers
#     plt.figtext(0.25, 0.9, 'Macro Average', fontweight='bold', ha='center')
#     plt.figtext(0.5, 0.9, 'Switch', fontweight='bold', ha='center')
#     plt.figtext(0.75, 0.9, 'Escalation', fontweight='bold', ha='center')

#     # Row Headers
#     plt.figtext(0.08, 0.78, 'F1', fontweight='bold', rotation='vertical')
#     plt.figtext(0.08, 0.48, 'Precision', fontweight='bold', rotation='vertical')
#     plt.figtext(0.08, 0.2, 'Recall', fontweight='bold', rotation='vertical')

#     # Shared 
#     # plt.figtext(0.5, 0.92, 'Reddit', ha='center')
#     plt.figtext(0.5, 0.08, 'Time Interval (days)', ha='center', fontweight='bold')

#     plt.show()
    
def subplot_3_by_3_averaged_across_seeds(df_meta_data, plot_std=True, min_time_interval_days=0,
                                             max_time_interval_days = 5,
                                             figsize=(15,15), shared_axes=True, save_fig=False, file_format = '.png'):
    
    # Change font size
    plt.rcParams.update({'font.size': 20})

    # sns.set(font_scale=5)  # crazy big
    
    # seed = 0
    # df_concat = pd.DataFrame()
    initialized = False
    for seed in df_meta_data['random_seed'].unique():
        df = df_meta_data[df_meta_data['random_seed'] == seed]
        
        t_intervals = list(np.array(list(range(max_time_interval_days*10+1))) * 0.1)
        x_max = t_intervals[-1]
        
        # # Get unique datasets from datadrame
        # datasets = list(df_meta_data['which_dataset'].unique())
        # which_dataset = datasets[0    
        # 
        df_filtered_time_intervals = return_dataframe_with_filtered_time_intervals(df, time_intervals_to_search=t_intervals, x_max=x_max)
        df = df_filtered_time_intervals
        df['random_seed'] = seed
        if initialized:
            df_concat = pd.concat([df_concat, df], axis=0)
            # df_concat = df
        else:
            df_concat = df
            initialized = True
    
    # Can concatenate with different seeds, then groupby model name and tau - and compute average scores.
    df = df_concat.groupby(['model_name', 'tau']).mean().reset_index().drop('random_seed', axis=1)
    
    # Get standard deviations, across seeds
    df_std = df_concat.groupby(['model_name', 'tau']).std().reset_index().drop('random_seed', axis=1)

    
    # Define hyper-parameters
    # all_model_names = ['HEAT', 'BiLSTM', '-BDR', '-MSD']
    all_model_names = ['BiLSTM-HEAT', '-BDR', '-MSD']

    
    row_length=3
    
    # Replace Name
    df['model_name'] = df['model_name'].replace('HEAT', 'BiLSTM-HEAT')

    
    fig, ax = plt.subplots(figsize=figsize, 
                        nrows=row_length, ncols=row_length,     # fix as above
                        # gridspec_kw=dict(hspace=0.4), # Much control of gridspec
                        sharex=shared_axes, sharey=shared_axes)

    for m, row in zip(['F1', 'P', 'R'], ax):
        for c, col in zip(['macro avg', 'S', 'E'], row):
            for model_name in all_model_names:
                
                metric = (c, m)
            
                y = df[df['model_name'] == model_name][metric]
                x = df[df['model_name'] == model_name]['tau']
                
                error = df_std[df_std['model_name'] == model_name][metric]

                col.plot(x, y, label=model_name)
                col.set_xlim(min_time_interval_days)
                
                # Fill with standard deviation, as error
                if plot_std:
                    col.fill_between(x, y-error, y+error,
                        alpha=0.1, linewidth=0)
                
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.legend()

    # Column Headers
    plt.figtext(0.25, 0.9, 'Macro Average', fontweight='bold', ha='center')
    plt.figtext(0.5, 0.9, 'Switch', fontweight='bold', ha='center')
    plt.figtext(0.75, 0.9, 'Escalation', fontweight='bold', ha='center')

    # Row Headers
    plt.figtext(0.01, 0.78, 'F1', fontweight='bold', rotation='vertical')
    plt.figtext(0.01, 0.48, 'Precision', fontweight='bold', rotation='vertical')
    plt.figtext(0.01, 0.2, 'Recall', fontweight='bold', rotation='vertical')

    # Shared 
    # plt.figtext(0.5, 0.92, 'Reddit', ha='center')
    plt.figtext(0.5, 0.01, 'Time Interval (days)', ha='center', fontweight='bold')

    # plt.tight_layout()
    
    if save_fig:
        str_format="%Y_%m_%d_%H_%M_%S"
        now = datetime.datetime.now().strftime(str_format)
        figure_name = 'pmocs_' + now

        plt.savefig(path_figure_save_path + figure_name + file_format, dpi=400)
    
    plt.show()


    return df

def return_final_reddit_talklife_results():
    """
    Returns a dictionary of 2 dataframes, containing the de-duplicated final 
    set of results on talklife and reddit.
    """

    # Get full dataframe, all results
    df = return_all_results_as_dataframe(
        include_meta_data=True,
        include_clpsych_results=True,
        round_to_sig_fig=None,
        deduplicate=False,
        remove_reddit_new=True,
        save_results=False,
    )

    full_df = df.copy()
    df = get_recent_renamed_results(df)

    df['random_seed'] = df['source_file'].apply(get_seed_from_classification_report_file_name)
    df = df.dropna()

    # Export individual results
    df_reddit = df[df['which_dataset'] == 'reddit']
    df_talklife = df[df['which_dataset'] == 'talklife']

    # Remove duplicates: deduplicate by seed number, dataset, model
    df_reddit = df_reddit.reset_index()
    df_reddit = df_reddit.loc[df_reddit[['random_seed', 'index', 'which_dataset']].drop_duplicates().index]
    df_reddit = df_reddit.set_index('index')

    df_reddit = df_reddit.reset_index().rename(columns={'index':'model_name'})
    df_reddit = df_reddit.sort_values('model_name')
    
    df_reddit = df_reddit.reset_index().drop('index', axis=1)

    # TalkLife
    df_talklife['folds'] = df_talklife['source_file'].apply(get_fold_number_from_file_name)

    # Get the folds as a int, not list
    df_talklife['folds'] = df_talklife['folds'].apply(lambda x: x[0])

    # Remove duplicates: deduplicate by seed number, dataset, model, and fold
    df_talklife = df_talklife.reset_index()
    df_talklife = df_talklife.loc[df_talklife[['random_seed', 'index', 'which_dataset', 'folds']].drop_duplicates().index]
    df_talklife = df_talklife.set_index('index')

    df_talklife = df_talklife.reset_index().rename(columns={'index':'model_name'})
    df_talklife = df_talklife.sort_values(['model_name', 'folds'])
    
    df_talklife = df_talklife.reset_index().drop('index', axis=1)
    
    # Get auxiliary file names also
    df_talklife['auxiliary_file_name'] = df_talklife['source_file'].apply(get_auxiliary_file_name_from_source_file_name)
    df_reddit['auxiliary_file_name'] = df_reddit['source_file'].apply(get_auxiliary_file_name_from_source_file_name)

    
    dfs = {'reddit': df_reddit, 'talklife': df_talklife}
    
    return dfs


def get_file_name_no_prefix_or_suffix(file_name, prefix='classification_reports_', suffix='.pickle'):
    name = file_name[len(prefix):-len(suffix)]
    
    return name

def get_auxiliary_file_name_from_source_file_name(file_name):
    name = get_file_name_no_prefix_or_suffix(file_name, prefix='classification_reports_', suffix='.pickle')
    ar_name = 'auxiliary_results_' + name + '.pickle'
    
    return ar_name

def create_data_frame_for_plotting_subplots(dfs, which_dataset='reddit', models=['HEAT', '-BDR', '-MSD']):

    seeds = [0, 1, 2]
    df_concat = pd.DataFrame()
    for random_seed in seeds:

        for model_name in models:
            df = join_predictions_with_metadata(dfs[which_dataset], model_name=model_name, random_seed=random_seed, which_dataset=which_dataset)
            df['model_name'] = model_name
            df['dataset'] = which_dataset

            if df_concat.empty:
                df_concat = df
            else:
                df_concat = pd.concat([df_concat, df], axis=0)
                
    df_concat = df_concat.reset_index().drop('index', axis=1)
    
    return df_concat