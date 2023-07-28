import sys
import os

sys.path.insert(0, "../../timeline_generation/")  # Adds higher directory to python modules path

from utils.io import data_handler

sys.path.insert(0, "../../global_utils/")  # Adds higher directory to python modules path
from notification_bot import send_email


def main():
    RedditDataset = data_handler.RedditDataset(include_embeddings=True, 
                                                save_processed_timelines=True, 
                                                load_timelines_from_saved=False)
    reddit_timelines = RedditDataset.timelines
    
    send_email(subject='[{}] Completed succesfully.'.format(os.path.basename(__file__)),
                message=str(reddit_timelines['timeline_id'].nunique()),
                receiving_email='angryasparagus@hotmail.com')

if __name__ == "__main__":
    try:
        main()
    except Exception as error_message:
        send_email(subject='[{}] Error, script failed.'.format(os.path.basename(__file__)),
            message=str(error_message),
            receiving_email='angryasparagus@hotmail.com')
