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
    experiment_name='all_variants_logreg_lstm'
    
    # Logreg variants
    

    # Specify models to use in experiments
    model_names = [
                # Logreg Models
                'logreg',
                
                # LogReg: HEAT
                'logreg_heat_past_only',  # Excludes current post in HEAT calculation
                'logreg_heat_past_present',  # Includes current post in HEAT

                # Concatenating present post raw rep., and future and past HEAT
                'logreg_concat_heat_past_present', 
                'logreg_concat_heat_past_present_future',
                'logreg_concat_exclude_present_in_heat_past_present', 
                'logreg_concat_exclude_present_in_heat_past_present_future',

                # LSTM Models
                "lstm_vanilla", 
                
                # LSTM: HEAT is applied over LSTM hidden states, rather than posts
                "lstm_heat_past_only",
                "lstm_heat_past_present",
                "lstm_heat_future_only",
                "lstm_heat_future_present",
                
                # LSTM: Concatenate past, future heat, and present post rep.atenate
                "lstm_concat_heat_past_present", 
                "lstm_concat_heat_past_present_future",
                "lstm_concat_heat_present_future",
                "lstm_concat_heat_past_future",
                
                # LSTM: Concat, but exclude current post in HEAT calculation
                "lstm_concat_exclude_present_in_heat_past_present",
                "lstm_concat_exclude_present_in_heat_past_present_future",
                "lstm_concat_exclude_present_in_heat_present_future",
                "lstm_concat_exclude_present_in_heat_past_future",

                # TODO: Note, below these have not been run yet - but are new models
                "lstm_vanilla_concat_present",
                "bilstm",
                "bilstm_concat_present",
                
                # Differencing Methods: Take cosine similarity, with HEAT
                'logreg_differencing_heat_normalized_concat_past_present', 
                'logreg_differencing_heat_unnormalized_concat_past_present',
                'logreg_differencing_heat_normalized_concat_past_present_future', 
                'logreg_differencing_heat_unnormalized_concat_past_present_future',
                'logreg_differencing_heat_normalized_concat_past_future',
                'logreg_differencing_heat_unnormalized_concat_past_future'
                   ]


    # hyperparams_to_search = {"learning_rate": [0.01, 0.1], 
    #                         "epochs": [1, 10],
    #                         'epsilon_prior': [0.001, 0.01, 0.1, 1, 10, 100], 
    #                         "beta_prior": [0.001, 0.01, 0.1, 1, 10, 100]}
    
    hyperparams_to_search = {"learning_rate": [0.0001, 0.001, 0.01, 0.1], 
                            "epochs": [100],
                            'epsilon_prior': [0.001, 0.01, 0.1, 1, 10, 100], 
                            "beta_prior": [0.001, 0.01, 0.1, 1, 10, 100],
                            "dropout": [0, 0.25, 0.50, 0.75],
                            "lstm1_hidden_dim": [64,128,256],
                            "lstm2_hidden_dim": [64,128,256],
                            # "batch_size": [16, 32, 64],
                            "number_of_lstm_layers": [2]
                            }

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
