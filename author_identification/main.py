import argparse
import time
import torch
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import LinearEmbeddingModel, entityEmbeddingModel, gloveEmbeddingModel
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from dataloader import AuthorDataset
from tqdm import tqdm
import unidecode

embedding_loc = 'data/glove.840B.300d.txt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

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
with open('data/embeddings.pkl', 'rb') as f:
    embeddings_index = pickle.load(f)


def update_embeddings(vocab):
    for token in vocab.itos:
        if token not in embeddings_index:
            embeddings_index[token] = np.random.normal(scale=0.6, size=(300,), )


def multiclass_logloss(predicted, actual, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def preprocess(s):
    x = unidecode.unidecode(s)
    x = ''.join(e for e in x.lower() if (e.isalnum() or e.isspace()))
    # x = ' '.join(lemmatizer.lemmatize(token) for token in x.split(" "))
    # x = ' '.join(lemmatizer.lemmatize(token, "v") for token in x.split(" "))
    x = ' '.join(word for word in x.split(" ") if not word in stop_words)
    return x


def get_tokenized_data(valid_ratio, X, y, X_test):
    X = list(map(lambda x: preprocess(x), X))
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
    train_null_idx = [i for i, x in enumerate(X_train) if (x == "") or (x == " ")]
    valid_null_idx = [i for i, x in enumerate(X_valid) if (x == "") or (x == " ")]
    print(train_null_idx)
    for idx in train_null_idx:
        del X_train[idx]
        del y_train[idx]
    for idx in valid_null_idx:
        del X_valid[idx]
        del y_valid[idx]
    update_embeddings(vocab)
    train_dataset = AuthorDataset(X_train, y_train, vocab, tokenizer, None, label_code, train=True,
                                  glove=embeddings_index)
    valid_dataset = AuthorDataset(X_valid, y_valid, vocab, tokenizer, None, label_code, train=True,
                                  glove=embeddings_index)

    return train_dataset, valid_dataset, vocab


def get_test_dataset(test_file_path, vocab):
    test_df = pd.read_csv(test_file_path)
    X_test = test_df['text']
    X_test = list(map(lambda x: preprocess(x), X_test))
    id = test_df['id']
    test_dataset = AuthorDataset(X_test, None, vocab, tokenizer, id, label_code, train=False, glove=embeddings_index)
    return test_dataset


def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_acc, total_loss, total_count = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        label, text = label.to(device), text.to(device)
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_loss += loss
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
    del text


def evaluate(model, dataloader, criterion):
    model.eval()
    total_acc, total_loss, total_count = 0, 0, 0
    predictions = []
    labels = []
    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            label, text = label.to(device), text.to(device)
            predicted = model(text)
            total_loss += criterion(predicted, label)
            total_acc += (predicted.argmax(1) == label).sum().item()
            total_count += label.size(0)
            pmf = F.softmax(predicted, dim=1)
            predictions.append(pmf)
            labels.append(label)
    print(pmf[0])
    predictions = np.reshape([item for sublist in predictions for item in sublist.tolist()], (-1, 3))
    labels = np.array([item for sublist in labels for item in sublist.tolist()])
    # metric = multiclass_logloss(predictions, labels)
    del text
    return total_acc / total_count, total_loss


def test(model, dataloader):
    model.eval()
    predictions, ids = [], []
    with torch.no_grad():
        for idx, (text, id) in enumerate(dataloader):
            text = text.to(device)
            predicted = model(text)
            predicted = F.softmax(predicted, dim=1)
            predictions.append(predicted)
            ids.append(id)
    # print([item for sublist in predictions for item in sublist.item()])
    predictions = np.reshape([item for sublist in predictions for item in sublist.tolist()], (-1, 3))
    ids = np.array([item for sublist in ids for item in sublist])
    eap, hpl, mws = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    data = {
        'id': ids,
        'EAP': eap,
        'HPL': hpl,
        'MWS': mws
    }
    df = pd.DataFrame(data=data)
    df.to_csv('outputs/result.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default='data/train.csv', help='train_file_path')
    parser.add_argument('--test_file_path', type=str, default='data/test.csv', help='test_file_path')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='proportion of validation samples')
    parser.add_argument('--embsize', type=int, default=37, help='embedding size')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.005489520509131124)
    parser.add_argument('--init_range', type=float, default=0.34261292447576064, help='range for weight initialization')

    args = parser.parse_args()

    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    valid_ratio = args.valid_ratio

    train_dataset, valid_dataset, vocab = get_dataset(train_file_path, test_file_path, valid_ratio)
    test_dataset = get_test_dataset(test_file_path, vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_dataset.char_level_collate, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False,
                              collate_fn=valid_dataset.char_level_collate)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
                             collate_fn=test_dataset.char_level_collate)

    for i, (label, text) in enumerate(train_loader):
        print(label)
        print(text.shape)
        print(text)
        break

    num_class = 3
    vocab_size = len(vocab)
    emsize = args.embsize
    init_range = args.init_range
    # model = LinearEmbeddingModel(vocab_size, num_class, num_layers=2, out_feats=[23, 102], dropouts=[0.50, 0.35],
    #                             embed_dim=args.embsize, init_range=args.init_range).to(device)

    # model = entityEmbeddingModel(vocab_size, num_class).to(device)
    model = gloveEmbeddingModel().to(device)
    epochs = args.epoch
    lr = args.learning_rate
    batch_size = args.batch_size
    total_acc = None
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, train_loader, optimizer, criterion, epoch)
        acc_val, loss_val = evaluate(model, valid_loader, criterion)
        if total_acc is not None and total_acc > acc_val:
            scheduler.step()
        else:
            total_acc = acc_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} | valid loss {:8.5f}'.format(epoch,
                                                                   time.time() - epoch_start_time,
                                                                   acc_val, loss_val))
        print('-' * 59)

    test(model, test_loader)
