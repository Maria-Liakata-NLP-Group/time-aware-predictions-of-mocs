
import sys

sys.path.insert(
    0, "../../predicting_mocs/"
)  # Adds higher directory to python modules path

# from utils.io import data_handler
from utils.visualize import aggregate_and_evaluate_results_from_multiple_experiments


file_names = [
    'auxiliary_results_talklife_2023_04_11_21_31_40____tl_fold_0_tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy_folds=[0]',
    'auxiliary_results_talklife_2023_04_11_21_32_12____tl_fold_1_tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy_folds=[1]',
    'auxiliary_results_talklife_2023_04_11_21_32_42____tl_fold_2_tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy_folds=[2]',
    'auxiliary_results_talklife_2023_04_11_21_29_07____tl_fold_3_tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy_folds=[3]',
    'auxiliary_results_talklife_2023_04_11_21_32_58____tl_fold_4_tanh_scaling_of_heat_bilstm_concat_bilstm_cross_entropy_folds=[4]'
]

scores = aggregate_and_evaluate_results_from_multiple_experiments(file_names)
print(scores)