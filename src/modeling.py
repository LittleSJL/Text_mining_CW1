import torch
import random
import torch.nn as nn
random.seed(1)
torch.manual_seed(1)


class BiLSTMClassifier(nn.ModuleList):
    def __init__(self, config, num_words, embedding_matrix=None):
        super().__init__()
        self.embedding_dim = int(config['PARAMETER']['embedding_dim'])
        self.lstm_layers = int(config['PARAMETER']['lstm_layers'])
        self.lstm_hidden_dim = int(config['PARAMETER']['lstm_hidden_dim'])

        # use pre-trained embedding
        if embedding_matrix is not None:
            if config['PARAMETER']['freeze'] == 'True':
                # freeze the embedding layer and the parameters will not update during training
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
            else:
                # fine-tune parameters of embedding layer during training
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            # randomly initialize the embedding layer (the parameters will update during training)
            self.embedding = nn.Embedding(num_words, self.embedding_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        self.fully_connected_layer = nn.Linear(in_features=self.lstm_hidden_dim * 2, out_features=50)

    def forward(self, x):
        h = torch.zeros((self.lstm_layers * 2, x.size(0), self.lstm_hidden_dim))  # initialize hidden state
        c = torch.zeros((self.lstm_layers * 2, x.size(0), self.lstm_hidden_dim))  # initialize cell state

        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = out[:, -1, :]  # get the hidden state of the last time step
        out = self.fully_connected_layer(out)

        return out


class BowClassifier(nn.Module):
    def __init__(self, config, num_words, embedding_matrix=None):
        super(BowClassifier, self).__init__()
        self.embedding_dim = int(config['PARAMETER']['embedding_dim'])

        if embedding_matrix is not None:
            if config['PARAMETER']['freeze'] == 'True':
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
            else:
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(num_words, self.embedding_dim)

        self.fully_connected_layer = nn.Linear(self.embedding_dim, 50)

    def forward(self, sentence):
        x = self.embedding(sentence)
        x = torch.mean(x, dim=1)
        x = self.fully_connected_layer(x)
        return x
