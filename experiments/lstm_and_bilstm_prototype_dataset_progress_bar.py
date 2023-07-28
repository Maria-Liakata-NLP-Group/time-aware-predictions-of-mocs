import os
import sys

from alive_progress import alive_bar

sys.path.insert(0, "../../timeline_generation/")  # Adds higher directory to python modules path
from utils.io import data_handler
from utils.io.my_pickler import my_pickler

sys.path.insert(0, "../../predicting_mocs/")  # Adds higher directory to python modules path
from models import heat
from pipelines.full_pipeline import (full_pipeline_for_multiple_models,
                                     full_pipeline_for_single_model)

sys.path.insert(0, "../../global_utils/")  # Adds higher directory to python modules path
from export_data import export_experiment_results, return_file_name
from notification_bot import send_email
from verbose import compute_number_of_iterations

# from verbose import progress_bar

def main():
    prototype = True
    verbose = False
    display_progress_bar = True
    experiment_name = return_file_name(os.path.basename(__file__))  # datetime_lstm_variants
    print_progress = True
    datasets = ['reddit', 'talklife']
    
    
    model_names = [
                # LSTM: Concat, but exclude current post in HEAT calculation
                "lstm_vanilla", 
                "bilstm",     
                ]

    hyperparams_to_search = {"learning_rate": [0.0001, 0.001, 0.01], 
                            "epochs": [100],
                            'epsilon_prior': [0.001, 0.01, 0.1, 1, 10, 100], 
                            "beta_prior": [0.001, 0.01, 0.1, 1, 10, 100],
                            "dropout": [0, 0.25, 0.50, 0.75],
                            "lstm1_hidden_dim": [64,128,256],
                            "lstm2_hidden_dim": [64,128,256],
                            # "batch_size": [16, 32, 64],
                            "number_of_lstm_layers": [2]
                            }
                            # 'patience': [5,10]}

    number_of_iterations = compute_number_of_iterations(hyperparams_to_search=hyperparams_to_search, number_of_datasets=2, models=model_names, number_of_folds=5)
    for i, which_data in enumerate(datasets):      # Carry out pipeline, for multiple models
        if print_progress:
            print("[{}/{}]\t({} / {})".format(i, len(datasets), which_data, datasets))
        
        with alive_bar(number_of_iterations) as progress_bar:
            classification_reports, results = full_pipeline_for_multiple_models(
                progress_bar=progress_bar,
                model_names=model_names,
                which_data=which_data,
                prototype=prototype,
                hyperparams_to_search=hyperparams_to_search,
                features=["sentence-bert"],
                target="label_3",
                perform_early_stopping=True,
                verbose=verbose,
                experiment_name=experiment_name,
                early_stopping_criterion='macro_f1',
                display_progress_bar = True,
                print_progress=True
            )
            # yield
            
            export_experiment_results(classification_reports, results, experiment_name=experiment_name, dataset=which_data, folder='results')

if __name__ == "__main__":
    
    main()
    # try:
    #     main()
    #     # with alive_bar(26, dual_line=True, title='Alphabet') as bar:
    #     # with alive_bar(1) as progress_bar:
    #     #     main(progress_bar)
    #     # progress_bar(main)
    # except Exception as error_message:
    #     send_email(subject='[{}] Error, script failed.'.format(os.path.basename(__file__)),
    #         message=str(error_message),
    #         receiving_email='angryasparagus@hotmail.com')
