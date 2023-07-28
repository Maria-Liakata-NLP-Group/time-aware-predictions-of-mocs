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

import sys

sys.path.insert(
    0, "../../timeline_generation/"
)  # Adds higher directory to python modules path

from utils.io import data_handler
from utils.io.my_pickler import my_pickler


device = "cuda:0" if cuda.is_available() else "cpu"


# Model Parameters. Same that were used for TalkLife.
BERT_params = {
    "dout": [0.25],  # [.25, .5]
    "lr": [1e-5, 3e-5],  # [1e-04, 1e-05], 5e-5, 3e-5.
    "max_len": [
        512
    ],  # Maximum content on Reddit was 16,581 characters. Mean is 573. Median is 230.
    "train_batch": 8,
    "valid_batch": 8,
    "epochs": 3,
}
my_ran_seed = 1
all_test_folds = [-1]  # Reddit, just use a single test fold rather than K-fold.
use_three_labels = True


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class BERTClass(torch.nn.Module):
    """
    Forward pass requires: (ids, mask, token_type_ids)
    """

    def __init__(self):
        super(BERTClass, self).__init__()
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


def train(epoch):
    model.train()
    with alive_bar(len(training_loader), title="Training...") as bar:  # Log progress
        for _, data in enumerate(training_loader, 0):
            bar()  # Log progress
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.long)

            # print('ids=\t{}'.format(ids))
            # print('mask=\t{}'.format(mask))
            # print('token_type_ids=\t{}'.format(token_type_ids))
            # print('targets=\t{}'.format(targets))

            outputs, representations = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            loss = loss_function(outputs, targets)
            if _ % 100 == 0:
                print(f"Epoch: {epoch}, Loss:  {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def validation(epoch):
    model.eval()
    val_targets, val_outputs, val_loss_total = [], [], 0.0
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.long)
            outputs, representations = model(ids, mask, token_type_ids)

            val_loss_total += loss_function(outputs, targets)
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return np.array(val_outputs), np.array(val_targets), val_loss_total.item()


def apply_to_test_set(epoch):
    model.eval()
    test_targets, test_outputs, test_loss_total = [], [], 0.0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.long)
            outputs, representations = model(ids, mask, token_type_ids)

            test_loss_total += loss_function(outputs, targets)
            test_targets.extend(targets.cpu().detach().numpy().tolist())
            test_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return np.array(test_outputs), np.array(test_targets), test_loss_total.item()


# def apply_to_full_original_dataset(epoch):
#     model.eval()
#     test_targets, test_outputs, test_loss_total = [], [], 0.0
#     with torch.no_grad():
#         for _, data in enumerate(full_original_loader, 0):
#             ids = data["ids"].to(device, dtype=torch.long)
#             mask = data["mask"].to(device, dtype=torch.long)
#             token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
#             targets = data["targets"].to(device, dtype=torch.long)
#             outputs = model(ids, mask, token_type_ids)

#             test_loss_total += loss_function(outputs, targets)
#             test_targets.extend(targets.cpu().detach().numpy().tolist())
#             test_outputs.extend(outputs.cpu().detach().numpy().tolist())
#     return np.array(test_outputs), np.array(test_targets), test_loss_total.item()


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


def predict(model):
    # Run forward pass
    with torch.no_grad():
        pred, representations = model(inputs)

    # Do something with pred
    pred = (
        pred.detach().cpu().numpy()
    )  # remove from computational graph to cpu and as numpy


def return_datasets_as_dataframes(
    df,
    which="reddit",
    val_fold=0,
    seed=1,
    use_three_labels=True,
):
    """
    Returns the training, validation, and test dataset in the same format as need for Adam's code.

    Needs ['post', 'label']
    """
    df = df.copy()

    # Select relevant data
    val_df = df[df["fold"] == val_fold]
    train_df = df[(df["fold"] != val_fold) & (df["train_or_test"] != "test")]
    test_df = df[df["train_or_test"] == "test"]

    # Post-processing
    val_df = post_process_dataframes(val_df, use_three_labels=use_three_labels)
    train_df = post_process_dataframes(train_df, use_three_labels=use_three_labels)
    test_df = post_process_dataframes(test_df, use_three_labels=use_three_labels)

    return train_df, val_df, test_df


# Fine_tune_bert_focal_model_on_reddit
set_seed(my_ran_seed)
myGenerator = torch.Generator()
myGenerator.manual_seed(my_ran_seed)
all_results = dict()
model_name = "bert_focal_loss"
dataset_name = "reddit"

all_val_folds = [
    0
]  # One fold for validation, 4 for training, and originally labelled test set used for test.

# Load full reddit dataset
RedditDataset = data_handler.RedditDataset(
    include_embeddings=True,
    save_processed_timelines=False,
    load_timelines_from_saved=True,
)
reddit_timelines = RedditDataset.timelines

for val_fold in all_val_folds:  # for each (test) fold
    print(
        "Training BERT (post) \tFold: "
        + str(val_fold + 1)
        + "/"
        + str(len(all_test_folds))
    )

    # Custom function to return these 3 dataframes
    train_dataset, val_dataset, test_df = return_datasets_as_dataframes(
        reddit_timelines,
        which="reddit",
        val_fold=val_fold,
        seed=my_ran_seed,
        use_three_labels=use_three_labels,
    )

    # param search grid
    best_eval_score = -1.0
    num_trial = 0
    best_lr = "nan"

    h = 0
    total_h = (
        len(BERT_params["max_len"]) * len(BERT_params["lr"]) * len(BERT_params["dout"])
    )
    for MAX_LEN in BERT_params["max_len"]:
        for LEARNING_RATE in BERT_params["lr"]:
            for DOUT in BERT_params["dout"]:
                h += 1
                print("Searching hyper-parameters:\t[{}/{}]".format(h, total_h))

                my_key = str(val_fold) + "_" + str(LEARNING_RATE) + "_" + str(DOUT)
                num_trial += 1

                TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS = (
                    BERT_params["train_batch"],
                    BERT_params["valid_batch"],
                    BERT_params["epochs"],
                )
                # print("Creating tokenizer...")
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                # print("Tokenizer created.")

                train_params = {
                    "batch_size": TRAIN_BATCH_SIZE,
                    "shuffle": False,  # Note this was previously True
                    "num_workers": 0,
                    "generator": myGenerator,
                    "worker_init_fn": 0,
                }
                val_params = {
                    "batch_size": VALID_BATCH_SIZE,
                    "shuffle": False,
                    "num_workers": 0,
                    "generator": myGenerator,
                    "worker_init_fn": 0,
                }
                test_params = {
                    "batch_size": VALID_BATCH_SIZE,
                    "shuffle": False,
                    "num_workers": 0,
                    "generator": myGenerator,
                    "worker_init_fn": 0,
                }

                # Apply pre-trained BERT on datasets
                training_set = CustomDataset(
                    train_dataset.post.values,
                    train_dataset.label.values,
                    tokenizer,
                    MAX_LEN,
                )
                validation_set = CustomDataset(
                    val_dataset.post.values,
                    val_dataset.label.values,
                    tokenizer,
                    MAX_LEN,
                )
                testing_set = CustomDataset(
                    test_df.post.values, test_df.label.values, tokenizer, MAX_LEN
                )

                # print("Creating data loaders...")
                training_loader = DataLoader(training_set, **train_params)
                validation_loader = DataLoader(validation_set, **val_params)
                testing_loader = DataLoader(testing_set, **test_params)
                # print("Data loaders created.")

                # Focal alpha:
                tr_ = np.array(train_dataset.label.values)
                class_proba = [
                    len(np.where(tr_ == 0)[0]) / len(tr_),
                    len(np.where(tr_ == 1)[0]) / len(tr_),
                    len(np.where(tr_ == 2)[0]) / len(tr_),
                ]
                inv_class_proba = np.sqrt(1 / np.array(class_proba))

                alphas_focal = list(inv_class_proba)
                # print("alphas_focal=\t{}".format(alphas_focal))

                # Defining the model
                # print("Instantiating the model...")
                model = BERTClass()
                # print("Moving model to device...")
                model.to(device)
                # print("Model ready on the device.")
                loss_function = FocalLoss(gamma=2, alpha=alphas_focal)  # Focal loss
                optimizer = torch.optim.Adam(
                    params=model.parameters(), lr=LEARNING_RATE
                )

                # Fine-tuning the model and validating
                # print("Fine-tuning and validating the model...")
                for epoch in range(EPOCHS):
                    # print("Training...")
                    train(epoch)

                    # print("Validating...")
                    # Make preds on validation set and measure loss/accuracy
                    preds, actual, current_eval_loss = validation(epoch)
                    preds = np.argmax(preds, axis=1)
                    f1_macro_dev = metrics.f1_score(
                        actual, preds, average="macro"
                    )  # adtsakal

                    print(
                        num_trial,
                        "Prev Best LR/loss:",
                        best_lr,
                        "\t",
                        best_eval_score,
                        "\n",
                        "Current Eval_loss:",
                        current_eval_loss,
                        "\tF1-Macro (dev):",
                        f1_macro_dev,
                    )

                    # Store best model
                    if f1_macro_dev > best_eval_score:  # if best model, save
                        best_lr = str(DOUT) + "_" + str(LEARNING_RATE)

                        preds, actual, _ = apply_to_test_set(epoch)
                        preds = np.argmax(preds, axis=1)

                        # Save best model
                        my_pickler(
                            "o",
                            "{}_fine_tuned_on_{}_val_fold={}".format(
                                model_name, dataset_name, val_fold
                            ),
                            model,
                            folder="models",
                        )

                        # # Save representations for associated posts, as dataframe
                        # my_pickler("o", model_name, folder="model_outputs")

                        print(
                            "F1-Macro (test):",
                            metrics.f1_score(actual, preds, average="macro"),
                            "\t(Saved)",
                        )
                        best_eval_score = f1_macro_dev
