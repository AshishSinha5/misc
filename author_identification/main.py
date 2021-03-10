import argparse
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import LinearEmbeddingModel
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from dataloader import AuthorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def get_tokenized_data(valid_ratio, X, y, X_test):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_ratio, stratify=y)
    counter = Counter()
    for line in X_train:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1)
    return list(X_train), list(X_valid), list(y_train), list(y_valid), vocab


def get_dataset(train_file_path, test_file_path, valid_ratio):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    X = train_df['text']
    y = train_df['author']
    X_test = test_df['text']
    X_train, X_valid, y_train, y_valid, vocab = get_tokenized_data(valid_ratio, X, y, X_test)

    train_dataset = AuthorDataset(X_train, y_train, vocab, tokenizer, label_code, train=True)
    valid_dataset = AuthorDataset(X_valid, y_valid, vocab, tokenizer, label_code, train=True)

    return train_dataset, valid_dataset, vocab


def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offset) in enumerate(dataloader):
        label, text, offset = label.to(device), text.to(device), offset.to(device)
        optimizer.zero_grad()
        predicted_label = model(text, offset)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(model, dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (label, text, offset) in enumerate(dataloader):
            label, text, offset = label.to(device), text.to(device), offset.to(device)
            predicted = model(text, offset)
            loss = criterion(predicted, label)
            total_acc += (predicted.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default='data/train.csv', help='train_file_path')
    parser.add_argument('--test_file_path', type=str, default='data/train.csv', help='test_file_path')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='proportion of validation samples')
    parser.add_argument('--embsize', type=int, default=32, help='embedding size')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5)
    parser.add_argument('--init_range', type=float, default=0.1, help='range for weight initialization')
    args = parser.parse_args()

    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    valid_ratio = args.valid_ratio

    train_dataset, valid_dataset, vocab = get_dataset(train_file_path, test_file_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_function)
    valid_loader = DataLoader(valid_dataset, batch_size=1000, shuffle=False,
                              collate_fn=valid_dataset.collate_function)


    num_class = 3
    vocab_size = len(vocab)
    emsize = args.embsize
    init_range = args.init_range
    model = LinearEmbeddingModel(vocab_size, emsize, num_class, init_range).to(device)

    epochs = args.epoch
    lr = args.learning_rate
    batch_size = args.batch_size
    total_acc = None
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, train_loader, optimizer, criterion, epoch)
        acc_val = evaluate(model, valid_loader, criterion)
        if total_acc is not None and total_acc > acc_val:
            scheduler.step()
        else:
            total_acc = acc_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               acc_val))
        print('-' * 59)
