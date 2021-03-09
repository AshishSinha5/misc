import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from dataloader import AuthorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

author_code = {
    'EAP': 'Edgar Allan Poe',
    'HPL': 'HP Lovecraft',
    'MWS': 'Mary Shelley'
}

label_code = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
}

tokenizer = get_tokenizer('basic_english')


def get_tokenized_data(args, X, y, X_test):
    valid_ratio = args.valid_ratio
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_ratio, stratify=y)
    counter = Counter()
    for line in X_train:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1)
    return list(X_train), list(X_valid), list(y_train), list(y_valid), vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default='data/train.csv', help='train_file_path')
    parser.add_argument('--test_file_path', type=str, default='data/train.csv', help='test_file_path')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='proportion of validation samples')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    args = parser.parse_args()

    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    X = train_df['text']
    y = train_df['author']
    X_test = test_df['text']
    X_train, X_valid, y_train, y_valid, vocab = get_tokenized_data(args, X, y, X_test)
    print(X_train[0])
    print(y_train[0])
    train_dataset = AuthorDataset(X_train, y_train, vocab, tokenizer, label_code, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_function)
    for x, y in train_loader:
        print(x[0])
        print(y)
        break
