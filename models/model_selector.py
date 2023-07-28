import sys

sys.path.insert(0, "../../predicting_mocs/")
# from utils.kfold import get_default_hyper_params

import torch
from models.logreg import (
    LogisticRegression,
    LogisticRegressionDifferencingHeat,
    LogisticRegressionDifferencingHeatConcat,
    LogisticRegressionHeat,
    LogisticRegressionHeatConcat,
)
from models.lstm import (
    BiLSTM,
    BiLSTMConcat,
    BiLSTMHeatAllowNegativeConcatBiLSTMSingleLayerWithLinearLayerTanh,
    BiLSTMHeatAllowNegativeNoSummationConcatBiLSTMSingleLayerWithLinearLayerTanh,
    BiLSTMHeatConcatBiLSTMSingleLayerWithLinearLayerTanh,
    BiLSTMHeatConcatBiLSTMTwoLayersWithLinearLayerTanh,
    BiLSTMHeatConcatPresentSingleLayer,
    BiLSTMHeatConcatPresentSingleLayerIncludeCurrentPost,
    BiLSTMHeatConcatPresentSingleLayerTemporalDirectionPastOnly,
    BiLSTMHeatConcatRawWithLinearLayerTanh,
    BiLSTMHeatConcatSingleLayerWithLinearLayerTanh,
    BiLSTMSingleLayer,
    HeatBackgroundIntensityMeanBilstm,
    HeatBiLSTMNoConcatenationOnlyHeat,
    HeatBiLSTMNoConcatenationOnlyHeatTanh,
    HeatDeltaReps,
    HeatLSTMNoBDR,
    HeatLSTMNoBDRConcatV,
    HeatNoMax,
    HeatNoMSD,
    HeatNoMSDConcatV,
    HeatRayleigh,
    HeatSoftplus,
    LearnableHeatAllowNegativeBiLSTMHeatConcatSingleLayerWithLinearLayerTanh,
    LearnableHeatLSTMHeatConcatSingleLayerWithLinearLayerTanh,
    LearnableHeatSigmoidBetaEpsilonLSTMHeatConcatSingleLayerWithLinearLayerTanh,
    LearnableHeatSoftplusBetaEpsilonAllowNegativeNoSummationLSTMHeatConcatSingleLayerWithLinearLayerTanh,
    LearnableHeatSoftplusBetaEpsilonLSTMHeatConcatSingleLayerWithLinearLayerTanh,
    LinguisticBiLSTMBiHeat,
    LinguisticConcatBiLSTMBiHeat,
    LSTMHeat,
    LSTMHeatConcat,
    LSTMHeatConcatLearnableWeights,
    LSTMHeatConcatPresentLSTMSingleLayer,
    LSTMHeatConcatSingleLayer,
    LSTMHeatConcatSingleLayerWithLinearLayerTanh,
    LSTMHeatWithoutHeatLayer,
    LSTMVanilla,
    LSTMVanillaConcat,
    LSTMVanillaSingleLayer,
    RamitHeat,
)

sys.path.insert(
    0, "../../global_utils/"
)  # Adds higher directory to python modules path
from global_parameters import device_to_use

device = torch.device(device_to_use if torch.cuda.is_available() else "cpu")

def model_selector(model_name="model_name", hyper_params=None, verbose=False):
    """
    Returns a given model, by an input string.

    Helpful for running within training
    or evaluation loops / pipelines - for not requiring to hardcode instantiating
    the model class. Instead just pass in the string, and will use the model name
    string to report the results, using the string in reporting.
    """
    # Select a model, given the string

    # LogReg models
    if model_name == "logreg":
        print("this is logreg")
        model = LogisticRegression()

    # LogReg HEAT
    elif model_name == "logreg_heat_past_only":
        model = LogisticRegressionHeat()
        model.exclude_current_post = True
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    elif model_name == "logreg_heat_past_present":
        model = LogisticRegressionHeat()
        model.exclude_current_post = False
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    elif model_name == "logreg_heat_future_only":
        model = LogisticRegressionHeat()
        model.exclude_current_post = True
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    elif model_name == "logreg_heat_future_present":
        model = LogisticRegressionHeat()
        model.exclude_current_post = False
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    # Concatenated: LogReg HEAT, concatenate past, present, future
    elif model_name == "logreg_concat_heat_past_present":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.concat_present = True
        model.fix_input_dim_and_linear_layer()
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    elif model_name == "logreg_concat_heat_past_present_future":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.fix_input_dim_and_linear_layer()

    elif model_name == "logreg_concat_heat_present_future":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.fix_input_dim_and_linear_layer()

    elif model_name == "logreg_concat_heat_past_future":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = False
        model.fix_input_dim_and_linear_layer()

    # Concatenated - exclude present in HEAT: LogReg HEAT, concatenate past, present, future
    elif model_name == "logreg_concat_exclude_present_in_heat_past_present":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.fix_input_dim_and_linear_layer()

    elif model_name == "logreg_concat_exclude_present_in_heat_past_present_future":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.fix_input_dim_and_linear_layer()

    elif model_name == "logreg_concat_exclude_present_in_heat_present_future":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.fix_input_dim_and_linear_layer()

    elif model_name == "logreg_concat_exclude_present_in_heat_past_future":
        model = LogisticRegressionHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = False
        model.fix_input_dim_and_linear_layer()

    # LSTM models
    elif model_name == "lstm_vanilla":
        model = LSTMVanilla()
        # model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
        # model.fix_dropout_layer()
    elif model_name == "lstm_vanilla_single_layer":
        model = LSTMVanillaSingleLayer()
        # model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
        # model.fix_dropout_layer()
    elif model_name == "lstm_vanilla_concat_present":
        model = LSTMVanillaConcat()
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name == "bilstm":
        model = BiLSTM()
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name == "bilstm_single_layer":
        model = BiLSTMSingleLayer()
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name == "bilstm_concat_present":
        model = BiLSTMConcat()
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    # LSTM HEAT
    elif model_name == "lstm_heat_past_only":
        model = LSTMHeat()
        model.exclude_current_post = True
        model.temporal_direction = "past"
        # model.epsilon = hyper_params["epsilon_prior"]
        # model.beta = hyper_params["beta_prior"]
        # model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_heat_past_present":
        model = LSTMHeat()
        model.exclude_current_post = False
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        # model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_heat_future_only":
        model = LSTMHeat()
        model.exclude_current_post = True
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        # model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_heat_future_present":
        model = LSTMHeat()
        model.exclude_current_post = False
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        # model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    # Concatenated: LogReg HEAT, concatenate past, present, future
    elif model_name == "lstm_concat_heat_past_present":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_heat_past_present_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name == "lstm_concat_heat_past_present_lstm_future_exclude_present":
        model = LSTMHeatConcatPresentLSTMSingleLayer()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_heat_present_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_heat_past_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = False
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    # Concatenated: LogReg HEAT, concatenate past, present, future
    elif model_name == "lstm_concat_heat_past_present":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_heat_past_present_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_heat_present_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_heat_past_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = False
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    # Concatenated - exclude present post in heat: concatenate past, present, future
    elif model_name == "lstm_concat_exclude_present_in_heat_past_present":
        model = LSTMHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_exclude_present_in_heat_past_present_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif (
        model_name == "lstm_concat_exclude_present_in_heat_past_present_future_1_layer"
    ):
        model = LSTMHeatConcatSingleLayer()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_exclude_present_in_heat_present_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "future"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "lstm_concat_exclude_present_in_heat_past_future":
        model = LSTMHeatConcat()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = False
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name == "LSTM_Heat_Without_Heat_Layer":
        model = LSTMHeatWithoutHeatLayer()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = False
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params

    # Learnable LSTM HEAT models
    elif (
        model_name
        == "lstm_learnable_concat_exclude_present_in_heat_past_present_future"
    ):
        model = LSTMHeatConcatLearnableWeights()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = torch.nn.Parameter(
            data=torch.tensor([hyper_params["epsilon_prior"]]), requires_grad=True
        )
        model.beta = torch.nn.Parameter(
            data=torch.tensor([hyper_params["beta_prior"]]), requires_grad=True
        )
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    # Differencing methods
    elif model_name == "logreg_differencing_heat_normalized":
        model = LogisticRegressionDifferencingHeat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = True
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    elif model_name == "logreg_differencing_heat_unnormalized":
        model = LogisticRegressionDifferencingHeat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = False
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    # Differencing methods
    elif model_name == "logreg_differencing_heat_normalized_concat_past_present":
        model = LogisticRegressionDifferencingHeatConcat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = True
        model.concat_present = True
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.fix_input_dim_and_linear_layer()
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    elif model_name == "logreg_differencing_heat_unnormalized_concat_past_present":
        model = LogisticRegressionDifferencingHeatConcat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = False
        model.concat_present = True
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.fix_input_dim_and_linear_layer()
        model.dropout = get_default_hyper_params(hyper_params, "dropout")

    elif model_name == "logreg_differencing_heat_normalized_concat_past_present_future":
        model = LogisticRegressionDifferencingHeatConcat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = True
        model.concat_present = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.fix_input_dim_and_linear_layer()

    elif (
        model_name == "logreg_differencing_heat_unnormalized_concat_past_present_future"
    ):
        model = LogisticRegressionDifferencingHeatConcat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = False
        model.concat_present = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.fix_input_dim_and_linear_layer()

    elif model_name == "logreg_differencing_heat_normalized_concat_past_future":
        model = LogisticRegressionDifferencingHeatConcat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = True
        model.concat_present = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.fix_input_dim_and_linear_layer()

    elif model_name == "logreg_differencing_heat_unnormalized_concat_past_future":
        model = LogisticRegressionDifferencingHeatConcat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = False
        model.concat_present = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.fix_input_dim_and_linear_layer()

    # Ramit HEAT models
    elif model_name == "ramit_lstm_heat":
        model = RamitHeat()
        model.exclude_current_post = False
        model.normalize_by_n_posts = False
        model.temporal_direction = "past"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.concat_present = True
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name == "ramit_lstm_heat_past_present_future_exclude_present":
        model = RamitHeat()
        model.exclude_current_post = True
        model.normalize_by_n_posts = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.concat_present = True
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "bilstm_heat_concat_present_single_layer":
        model = BiLSTMHeatConcatPresentSingleLayer()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "bilstm_heat_concat_present_single_layer_temporal_direction_past_only"
    ):
        model = BiLSTMHeatConcatPresentSingleLayerTemporalDirectionPastOnly()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        # model.fix_input_dim_and_linear_layer()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "bilstm_heat_concat_present_single_layer_include_current_post"
    ):
        model = BiLSTMHeatConcatPresentSingleLayerIncludeCurrentPost()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "lstm_heat_concat_single_layer_with_linear_layer_tanh":
        model = LSTMHeatConcatSingleLayerWithLinearLayerTanh()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    elif model_name.lower() == "bilstm_heat_concat_single_layer_with_linear_layer_tanh":
        model = BiLSTMHeatConcatSingleLayerWithLinearLayerTanh()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "bilstm_heat_concat_single_layer_with_linear_layer_tanh_include_current_post"
    ):
        model = BiLSTMHeatConcatSingleLayerWithLinearLayerTanh()
        model.exclude_current_post = False
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh"
    ):
        model = BiLSTMHeatConcatBiLSTMSingleLayerWithLinearLayerTanh()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "bilstm_heat_concat_raw_with_linear_layer_tanh":
        model = BiLSTMHeatConcatRawWithLinearLayerTanh()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower() == "2_layer_bilstm_heat_concat_bilstm_with_linear_layer_tanh"
    ):
        model = BiLSTMHeatConcatBiLSTMTwoLayersWithLinearLayerTanh()
        model.exclude_current_post = True
        model.temporal_direction = "both"
        model.epsilon = hyper_params["epsilon_prior"]
        model.beta = hyper_params["beta_prior"]
        model.dropout = get_default_hyper_params(hyper_params, "dropout")
        model.concat_present = True
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "learnable_heat_lstm_heat_concat_single_layer_with_linear_layer_tanh"
    ):
        model = LearnableHeatLSTMHeatConcatSingleLayerWithLinearLayerTanh()

        # Note, that all hyper-parameters are now applied in apply_hyperparameters
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "learnable_heat_sigmoid_eb_lstm_heat_concat_single_layer_with_linear_layer_tanh"
    ):
        model = (
            LearnableHeatSigmoidBetaEpsilonLSTMHeatConcatSingleLayerWithLinearLayerTanh()
        )
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "learnable_heat_softplus_eb_lstm_heat_concat_single_layer_with_linear_layer_tanh"
    ):
        model = (
            LearnableHeatSoftplusBetaEpsilonLSTMHeatConcatSingleLayerWithLinearLayerTanh()
        )
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh_no_summation"
    ):
        model = BiLSTMHeatNoSummationConcatBiLSTMSingleLayerWithLinearLayerTanh()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh_allow_negative_decayed_x"
    ):
        model = BiLSTMHeatAllowNegativeConcatBiLSTMSingleLayerWithLinearLayerTanh()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "bilstm_heat_concat_bilstm_single_layer_with_linear_layer_tanh_allow_negative_decayed_x_no_summation"
    ):
        model = (
            BiLSTMHeatAllowNegativeNoSummationConcatBiLSTMSingleLayerWithLinearLayerTanh()
        )
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "learnable_heat_softplus_allow_negative_no_summation_lstm_heat_concat_single_layer_with_linear_layer_tanh"
    ):
        model = (
            LearnableHeatSoftplusBetaEpsilonAllowNegativeNoSummationLSTMHeatConcatSingleLayerWithLinearLayerTanh()
        )
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif (
        model_name.lower()
        == "learnable_heat_allow_negative_bilstm_heat_concat_single_layer_with_linear_layer_tanh"
    ):
        model = (
            LearnableHeatAllowNegativeBiLSTMHeatConcatSingleLayerWithLinearLayerTanh()
        )
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_bilstm_no_concatenation_only_heat_tanh":
        model = HeatBiLSTMNoConcatenationOnlyHeatTanh()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_bilstm_no_concatenation_only_heat":
        model = HeatBiLSTMNoConcatenationOnlyHeat()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_no_msd":
        model = HeatNoMSD()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_no_msd_concat_v":
        model = HeatNoMSDConcatV()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_lstm_no_bdr":
        model = HeatLSTMNoBDR()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_lstm_no_bdr_concat_h":
        model = HeatLSTMNoBDRConcatV()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "linguistic_bilstm_biheat":
        model = LinguisticBiLSTMBiHeat()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "linguistic_concat_bilstm_biheat_concat_h":
        model = LinguisticConcatBiLSTMBiHeat()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()            
    
    # Post 1st submission models
    elif model_name.lower() == "heat_background_intensity_mean_bilstm":
        model = HeatBackgroundIntensityMeanBilstm()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_delta_reps_bilstm":
        model = HeatDeltaReps()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_no_max_bilstm":
        model = HeatNoMax()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_rayleigh_bilstm":
        model = HeatRayleigh()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()
    elif model_name.lower() == "heat_softplus_bilstm":
        model = HeatSoftplus()
        model.hyper_params = hyper_params
        model.apply_hyperparameters()

    # If selected model not defined in function, throw error
    else:
        print("Error: `{}` model not defined in `model_selector`.".format(model_name))

    # Print info, if desired
    if verbose:
        print("`{}` model instantiated.".format(model_name))

    # Move parameters to GPU
    model.to(device)

    return model


def check_if_model_is_timeline_sensitive(model_name="model_name"):
    """
    Returns whether the model is timeline sensitive. If so,
    then this information can be used to process the dataset
    accordingly.
    """
    timeline_sensitive_keywords = ["lstm", "heat"]

    if any(keyword in model_name.lower() for keyword in timeline_sensitive_keywords):
        is_time_sensitive = True
    else:
        is_time_sensitive = False

    return is_time_sensitive


def check_if_model_is_time_aware(model_name="model_name"):
    """
    Returns whether the model is sensitive to the timestamps of the model
    inputs. If so, then this information can be used to process the dataset
    accordingly (e.g. appending timestamps at the end of the input x features
    in the dataloader, which will then be sliced out to differentiate
    embeddings/ training features and auxiliary timestamp information).
    """
    time_aware_keywords = ["heat"]

    if any(keyword in model_name.lower() for keyword in time_aware_keywords):
        is_time_aware = True
    else:
        is_time_aware = False

    return is_time_aware


def check_if_model_uses_this_hyper_parameter(
    model_name, hyper_parameter_name, loss_name="cross_entropy"
):
    """
    Returns True if the input `model_name` makes use of the input
    `hyper_parameter_name`.

    Will be handy for checking whether to continue the grid-search for the
    given hyper-parameter or not.
    """

    # Default hyper-parameters, which all models generally use
    default_hyper_parameters = [
        "learning_rate",
        "epochs",
        "batch_size",
        "dropout",
        "patience",
    ]
    possible_hyper_parameters = default_hyper_parameters

    # Models which use HEAT should also have epsilon and beta
    if "heat" in model_name.lower():
        possible_hyper_parameters += ["epsilon_prior", "beta_prior"]

    # LSTM models
    if "lstm" in model_name.lower():
        possible_hyper_parameters += [
            "number_of_lstm_layers",
            "lstm_hidden_dim",
            "lstm1_hidden_dim",
            "lstm2_hidden_dim",
            "lstm_hidden_dim_global",
        ]

    # Loss functions
    if "focal" in loss_name.lower():
        possible_hyper_parameters += ["gamma"]

    if "class_balanced" in loss_name.lower():
        possible_hyper_parameters += ["beta_cb"]

    model_accepts_this_hyper_parameter = (
        hyper_parameter_name in possible_hyper_parameters
    )

    return model_accepts_this_hyper_parameter


def return_only_valid_hyper_parameters_for_model(
    model_name, hyper_parameters, loss_name="cross_entropy"
):
    """
    Takes an input `model_name` and dictionary `dict_hyper_parameters_to_search`,
    and removes keys from the dictionary if they are not valid for the
    given model name.
    """

    # e.g. learning_rate, epochs, epsilon_prior, etc.
    hyper_param_types = list(hyper_parameters.keys())

    # loop over each hyper-parameter type in dictionary
    for h in hyper_param_types:

        # False if model does not accept this hyper-parameter
        model_accepts_this_hyper_param = check_if_model_uses_this_hyper_parameter(
            model_name, h, loss_name=loss_name
        )

        # Remove the hyper-parameter key, if this model doesn't use it
        if model_accepts_this_hyper_param == False:
            hyper_parameters.pop(h, None)

        # Ensure that single layer models only have a single hidden dimension
        if ("1_layer" in model_name) & (h == "lstm2_hidden_dim"):
            hyper_parameters.pop(h, None)

    return hyper_parameters


def get_default_hyper_params(dict_hyper_params, which="learning_rate"):
    """
    Returns the specified hyperparameters, and if None are returned,
    then returns from a default list.
    """
    h = dict_hyper_params.get(which)

    if h == None:
        if which == "learning_rate":
            h = 0.01
        elif which == "epochs":
            h = 1
        elif which == "dropout":
            h = 0  # Does not apply dropout, by default

    return h
