from torch import nn
import torch


class LinearEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, num_class, num_layers, out_feats=None, dropouts=None, embed_dim=32, init_range=0.1):
        super(LinearEmbeddingModel, self).__init__()
        self.num_layers = num_layers
        self.out_feats = out_feats
        self.dropouts = dropouts
        self.embed_dim = embed_dim
        self.init_range = init_range
        self.embedding = nn.EmbeddingBag(vocab_size, self.embed_dim)
        # self.fc = nn.Linear(embed_dim, num_class)
        self.layers = []
        in_feat = self.embed_dim
        for i in range(self.num_layers):
            out_feat = self.out_feats[i]
            self.layers.append(nn.Linear(in_feat, out_feat))
            p = self.dropouts[i]
            self.layers.append(nn.Dropout(p))
            in_feat = out_feat
        if len(self.layers):
            self.layers = nn.ModuleList(self.layers)
        self.bn = nn.BatchNorm1d(in_feat)
        self.op = nn.Linear(in_feat, num_class)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
        # self.fc.weight.data.data.uniform_(-self.init_range, self.init_range)
        # self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        for layer in self.layers:
            x = layer(x)
        x = self.bn(x)
        x = self.op(x)
        return x


class entityEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, num_class, embed_dim=32, hidden_dim=16, bidirectional=True):
        super(entityEmbeddingModel, self).__init__()
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.op = nn.Linear(2*hidden_dim, num_class)
        else:
            self.op = nn.Linear(hidden_dim, num_class)

    def forward(self, text):
        x = self.embedding(text)
        lstm_op, (ht, ct) = self.lstm(x)
        if self.bidirectional:
            ht = torch.unsqueeze(torch.cat([ht[i, :, :] for i in range(2)], dim=-1), dim=0)
        x = self.op(ht[-1])
        return x


class gloveEmbeddingModel(nn.Module):
    def __init__(self, embed_dim=300, num_class=3, hidden_dim=100, bidirectional=True):
        super(gloveEmbeddingModel, self).__init__()
        self.input_dim = embed_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional, dropout=0.3)
        if self.bidirectional:
            self.fc1 = nn.Linear(hidden_dim*2, 256)
        else:
            self.fc1 = nn.Linear(hidden_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.6)
        self.op = nn.Linear(256, num_class)

    def forward(self, text):
        # text = torch.unsqueeze(text, -1)
        lstm_op, (ht, ct) = self.lstm(text)
        if self.bidirectional:
            ht = torch.unsqueeze(torch.cat([ht[i, :, :] for i in range(2)], dim=-1), dim=0)
        x = self.fc1(ht[-1])
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.op(x)
        return x




