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
    def __init__(self):
        super(gloveEmbeddingModel, self).__init__()
        pass

    def forward(self, x):
        return x

