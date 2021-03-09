import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(s):
    return ''.join(e for e in s.lower() if (e.isalnum() or e.isspace()))


class AuthorDataset(Dataset):

    def __init__(self, X, y, vocab, tokenizer, label_code=None, train=True):
        self.X = X
        self.y = y
        self.train = train
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.label_code = label_code

    def __len__(self):
        return len(self.X)

    def text_pipeline(self, line):
        return [self.vocab[tokens] for tokens in self.tokenizer(line)]

    def __getitem__(self, index):
        x = torch.tensor(self.text_pipeline(preprocess(self.X[index])))
        if self.train:
            y = self.label_code[self.y[index]]
            return x, y
        return x

    def collate_function(self, data):
        if self.train:
            (x, y) = zip(*data)
            x = list(x)
            return x, torch.tensor(y)




