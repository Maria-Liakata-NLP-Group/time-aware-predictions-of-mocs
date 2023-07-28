import numpy as np
from sentence_transformers import SentenceTransformer

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from numpy.lib.function_base import average
from sklearn import metrics
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from alive_progress import alive_bar

import pickle
import pandas as pd
from os import walk
import torch

import sys

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

# from utils.io import data_handler
from utils.io.my_pickler import my_pickler

# sys.path.insert(0, "../../predicting_mocs/")  # Adds higher directory to python modules path
# from experiments.fine_tune_bert_focal import post_process_dataframes

sentence_bert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")


def text_vectorizer(post, embedding_type="sentence-bert", apply_preprocessing=False):
    if apply_preprocessing:
        post = pre_process(post)

    # Sentence Bert
    if embedding_type == "sentence-bert":
        vectorised_post = sentence_bert_model.encode([post]).reshape(1, -1)
    else:
        print("Error, {} is not a vectorizer we have code for.".format(embedding_type))

    return vectorised_post


def pre_process(text):
    text = text.lower()
    return text

class BERTClass(torch.nn.Module):
    """
    Forward pass requires: (ids, mask, token_type_ids)
    """
    
    def __init__(self):
        super(BERTClass, self).__init__()
        use_three_labels=True
        DOUT=0.25
        self.l1 = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.l2 = torch.nn.Dropout(DOUT)
        if use_three_labels:
            self.l3 = torch.nn.Linear(768, 3)
        else:
            self.l3 = torch.nn.Linear(768, 5)

    def forward(self, ids, mask, token_type_ids):
        _, representations = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        output_2 = self.l2(representations)  # Dropout layer
        output = self.l3(output_2)  # Linear Layer
        
        return output, representations  # output_1 is the embeddings


class CustomDataset(Dataset):
    def __init__(self, posts, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.post_texts = posts
        self.targets = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.post_texts)

    def __getitem__(self, index):
        post_text = str(self.post_texts[index])
        post_text = " ".join(post_text.split())

        inputs = self.tokenizer.encode_plus(
            post_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.long),
        }

    def forward(self, ids, mask, token_type_ids):
        _, representations = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        output_2 = self.l2(representations)  # Dropout layer
        output = self.l3(output_2)  # Linear Layer

        return output, representations  # output_1 is the embeddings


def post_process_dataframes(df, use_three_labels=True):
    # Post-processing
    if use_three_labels:
        df = df[["content", "label_3"]]  # Only get relevant columns
    df = df.rename(columns={"content": "post", "label_3": "label"})
    df = df.reset_index(drop=True)
    df["label"] = df["label"].apply(
        convert_labels_to_categorical
    )  # Convert to categorical

    return df


def convert_labels_to_categorical(labels, three_labels=True):
    """
    Converting string labels to their categorical version.
    """
    if three_labels:
        vals = {"0": 0, "E": 1, "S": 2}
    else:
        vals = {"0": 0, "IE": 1, "IEP": 2, "IS": 3, "ISB": 4}
    return np.array([vals[k] for k in labels])


def apply_to_full_original_dataset(model, full_data_loader):
    model.eval()
    test_targets, test_outputs, test_loss_total = [], [], 0.0
    with torch.no_grad():
        with alive_bar(len(full_data_loader), title="Predicting...") as bar:
            for _, data in enumerate(full_data_loader, 0):
                bar()
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data["mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)

                # test_loss_total += loss_function(outputs, targets)
                test_targets.extend(targets.cpu().detach().numpy().tolist())
                test_outputs.extend(outputs.cpu().detach().numpy().tolist())

    return (
        outputs,
        np.array(test_outputs),
        np.array(test_targets),
    )  # , test_loss_total.item()


def apply_fine_tuned_model_to_dataset(
    which_dataset="reddit", val_fold=0, model_name="bert_focal_loss"
):

    # Load Reddit dataset
    df_reddit_timelines = my_pickler(
        "i", "reddit_timelines_sentence-bert", folder="datasets"
    )
    df = df_reddit_timelines.copy()
    df = post_process_dataframes(df, use_three_labels=True)

    # Load model
    model = BERTClass()
    model.load_state_dict(
        my_pickler(
            "i",
            "{}_fine_tuned_on_{}_val_fold={}".format(
                model_name, which_dataset, val_fold
            ),
            folder="models",
        )
    )
    # model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    my_ran_seed = 1
    myGenerator = torch.Generator()
    myGenerator.manual_seed(my_ran_seed)

    full_dataset = CustomDataset(
        df.post.values,
        df.label.values,
        tokenizer,
        512,
    )
    full_data_params = train_params = {
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 0,
        "generator": myGenerator,
        "worker_init_fn": 0,
    }

    full_data_loader = DataLoader(full_dataset, **full_data_params)

    representations, test_outputs, targets = apply_to_full_original_dataset(
        model, full_data_loader
    )

    df_reddit_timelines["bert_focal_loss"] = representations
    df_reddit_timelines["bert_focal_loss"] = df_reddit_timelines[
        "bert_focal_loss"
    ].apply(lambda x: np.array(x).reshape(1, -1))
    my_pickler(
        "o", "reddit_timelines_all_embeddings", df_reddit_timelines, folder="datasets"
    )
