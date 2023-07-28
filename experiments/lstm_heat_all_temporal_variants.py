import os
import sys

sys.path.insert(0, "../../timeline_generation/")  # Adds higher directory to python modules path
from utils.io import data_handler
from utils.io.my_pickler import my_pickler

sys.path.insert(0, "../../predicting_mocs/")  # Adds higher directory to python modules path
from models import heat
from pipelines.full_pipeline import (full_pipeline_for_multiple_models,
                                     full_pipeline_for_single_model)

sys.path.insert(0, "../../global_utils/")  # Adds higher directory to python modules path
from export_data import export_experiment_results
from notification_bot import send_email


def main():
    experiment_name='lstm_multiple_heat_models'

    # Specify models to use in experiments
    model_names = ["lstm_vanilla", 
                   "lstm_heat_past_only",
                   "lstm_heat_past_only",
                   "lstm_heat_past_present",
                   "lstm_heat_future_only",
                   "lstm_heat_future_present",
                   
                   "lstm_concat_heat_past_present", 
                   "lstm_concat_heat_past_present_future",
                   "lstm_concat_heat_present_future",
                   "lstm_concat_heat_past_future",
                   "lstm_concat_exclude_present_in_heat_past_present_future",
                   "lstm_concat_exclude_present_in_heat_present_future",
                   "lstm_concat_exclude_present_in_heat_past_future",
                   ]


    hyperparams_to_search = {"learning_rate": [0.01, 0.1], 
                            "epochs": [1, 10],
                            'epsilon_prior': [0.001, 0.01, 0.1, 1, 10, 100], 
                            "beta_prior": [0.001, 0.01, 0.1, 1, 10, 100]}

    for which_data in ['reddit', 'talklife']:

        # Carry out pipeline, for multiple models
        classification_reports, results = full_pipeline_for_multiple_models(
            model_names=model_names,
            which_data=which_data,
            prototype=False,
            hyperparams_to_search=hyperparams_to_search,
            features=["sentence-bert"],
            target="label_3",
        )
        
        export_experiment_results(classification_reports, results, experiment_name=experiment_name, dataset=which_data, folder='results')

if __name__ == "__main__":
    try:
        main()
    except Exception as error_message:
        send_email(subject='[{}] Error, script failed.'.format(os.path.basename(__file__)),
            message=str(error_message),
            receiving_email='angryasparagus@hotmail.com')
