import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from sklearn import metrics
import random
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import BertTokenizer

from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'

import data_handler
from _utils import FOLD_to_TIMELINE, FOLDER_models, BERT_params



class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


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
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }
        
        
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(DOUT)
        if use_three_labels:
            self.l3 = torch.nn.Linear(768, 3)
        else:
            self.l3 = torch.nn.Linear(768, 5)
            
    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
        

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        if _%100==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def validation(epoch):
    model.eval() 
    val_targets, val_outputs, val_loss_total = [], [], 0.0
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            
            val_loss_total+=loss_function(outputs, targets)
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return np.array(val_outputs), np.array(val_targets), val_loss_total.item()


def apply_to_test_set(epoch):
    model.eval() 
    test_targets, test_outputs, test_loss_total = [], [], 0.0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            
            test_loss_total+=loss_function(outputs, targets)
            test_targets.extend(targets.cpu().detach().numpy().tolist())
            test_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return np.array(test_outputs), np.array(test_targets), test_loss_total.item()


def convert_labels_to_categorical(labels, three_labels=True):
    '''
    Converting string labels to their categorical version.
    '''
    if three_labels:
        vals = {'0':0, 'IE':1, 'IS':2}
    else:
        vals = {'0':0, 'IE':1, 'IEP':2, 'IS':3, 'ISB':4}    
    return np.array([vals[k] for k in labels])


def convert_categorical_to_labels(categories, three_labels=True):
    '''
    Converting categorical predictions to their actual (string) class.
    '''
    if three_labels:
        vals = ['0', 'IE', 'IS']
    else:
        vals = ['0', 'IE', 'IEP', 'IS', 'ISB']
    return np.array([vals[int(k)] for k in categories])        



