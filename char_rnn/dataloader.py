import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(s):
    return ''.join(e for e in s.lower() if (e.isalnum() or e.isspace()))


class AuthorDataset(Dataset):

    def __init__(self, X, y, vocab, tokenizer, train=True ):
        self.X = X
        self.y = y
        self.train = train
        self.text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = []
        y = []
        for idx in index:
            pass








