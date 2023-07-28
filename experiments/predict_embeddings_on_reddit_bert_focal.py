import numpy as np
import pandas as pd
import torch
import transformers
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from alive_progress import alive_bar

import pandas as pd
import torch

import sys

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

# from utils.io import data_handler
from utils.io.my_pickler import my_pickler

# sys.path.insert(0, "../../predicting_mocs/")  # Adds higher directory to python modules path
# from experiments.fine_tune_bert_focal import post_process_dataframes


device = "cuda:0" if cuda.is_available() else "cpu"


class BERTClass(torch.nn.Module):
    """
    Forward pass requires: (ids, mask, token_type_ids)
    """

    def __init__(self):
        DOUT = 0.25
        use_three_labels = True

        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(
            "bert-base-uncased"
        )  # TODO: This is causing problems, killing kernel
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
        # vals = {"0": 0, "E": 1, "S": 2}
        vals = {"S": 0, "E": 1, "0": 2}
    else:
        vals = {"0": 0, "IE": 1, "IEP": 2, "IS": 3, "ISB": 4}
    return np.array([vals[k] for k in labels])


def convert_categorical_to_labels(categories, three_labels=True):
    """
    Converting categorical predictions to their actual (string) class.
    """
    if three_labels:
        # vals = ["0", "E", "S"]
        vals = ["S", "E", "0"]
    else:
        vals = ["0", "IE", "IEP", "IS", "ISB"]
    return np.array([vals[int(k)] for k in categories])


def apply_to_full_original_dataset(model, full_data_loader):
    # print('a')
    model.eval()
    test_targets, test_outputs, test_loss_total, all_representations = [], [], 0.0, []
    # print('b')
    with torch.no_grad():
        # print('c')
        with alive_bar(len(full_data_loader), title="Predicting...") as bar:
            # print('d')
            for _, data in enumerate(full_data_loader, 0):
                # print('e')
                # print(_)
                bar()
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data["mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.long)
                outputs, representations = model(ids, mask, token_type_ids)

                # test_loss_total += loss_function(outputs, targets)
                all_representations.extend(
                    representations.cpu().detach().numpy().tolist()
                )
                test_targets.extend(targets.cpu().detach().numpy().tolist())
                test_outputs.extend(outputs.cpu().detach().numpy().tolist())

    return (
        all_representations,
        np.array(test_outputs),
        np.array(test_targets),
    )  # , test_loss_total.item()


model = BERTClass()
model = my_pickler(
    "i", "bert_focal_loss_fine_tuned_on_reddit_val_fold=0", folder="models"
)

# Load Reddit dataset
df_reddit_timelines = my_pickler(
    "i", "reddit_timelines_sentence-bert", folder="datasets"
)
df = df_reddit_timelines.copy()
df = post_process_dataframes(df, use_three_labels=True)

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
full_data_params = {
    "batch_size": 8,
    "shuffle": False,
    "num_workers": 0,
    "generator": myGenerator,
    "worker_init_fn": 0,
}

full_data_loader = DataLoader(full_dataset, **full_data_params)

representations, b, c = apply_to_full_original_dataset(model, full_data_loader)

my_pickler(
    "o", "bert_focal_loss_outputs_on_reddit", [representations, b, c], folder="results"
)

# Create embeddings on dataframe
outputs = my_pickler("i", "bert_focal_loss_outputs_on_reddit", folder="results")

df_reddit = df_reddit_timelines = my_pickler(
    "i", "reddit_timelines_sentence-bert", folder="datasets"
)

representations = outputs[0]
df_reddit["bert_focal_loss"] = representations
df_reddit["bert_focal_loss"] = df_reddit["bert_focal_loss"].apply(
    lambda x: np.array(x).reshape(1, -1)
)
my_pickler("o", "reddit_timelines_bert_focal_loss", df_reddit, folder="datasets")
