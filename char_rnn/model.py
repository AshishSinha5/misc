from torch import nn


class LinearEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, init_range):
        super(LinearEmbeddingModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights(init_range)

    def init_weights(self, init_range):
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embed = self.embedding(text, offsets)
        return self.fc(embed)

