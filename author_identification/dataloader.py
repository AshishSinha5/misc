import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AuthorDataset(Dataset):

    def __init__(self, X, y, vocab, tokenizer, id=None, label_code=None, train=True):
        self.X = X
        self.y = y
        self.ID = id
        self.train = train
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.label_code = label_code

    def __len__(self):
        return len(self.X)

    def text_pipeline(self, line):
        return [self.vocab[tokens] for tokens in self.tokenizer(line)]

    def __getitem__(self, index):
        x = torch.tensor(self.text_pipeline(self.X[index]))
        if self.train:
            y = self.label_code[self.y[index]]
            return x, y
        else:
            id = self.ID[index]
            return x, id

    def collate_function(self, batch):
        offsets = [0]
        if self.train:
            (text_list, label_list) = zip(*batch)
            text_list = torch.cat(text_list)
            label_list = torch.tensor(label_list)
            for _text, _ in batch:
                offsets.append(_text.size(0))
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            return label_list.to(device), text_list.to(device), offsets.to(device)
        else:
            (text_list, id) = zip(*batch)
            text_list = torch.cat(text_list)
            for _text, _ in batch:
                offsets.append(_text.size(0))
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            return text_list.to(device), offsets.to(device), list(id)

    def char_level_collate(self, batch):
        if self.train:
            (text_list, label_list) = zip(*batch)
            padded_tensors = torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in text_list], batch_first=True)
            label_list = torch.tensor(label_list)
            return label_list.to(device), padded_tensors.to(device)

        else:
            (text_list, id) = zip(*batch)
            padded_tensors = torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in text_list], batch_first=True)
            return text_list.to(device).long(), list(id)