import torch
import random
import torch.nn as nn
random.seed(1)
torch.manual_seed(1)


class BowClassifier(nn.Module):
    def __init__(self, config, num_words, embedding_matrix=None, device=None):
        super(BowClassifier, self).__init__()
        self.device = device
        self.embedding_dim = int(config['PARAMETER']['embedding_dim'])

        # embedding layer
        if embedding_matrix is not None:
            if config['PARAMETER']['freeze'] == 'True':
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
            else:
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(num_words, self.embedding_dim)

        # classification layer to generate the final classification
        self.fully_connected_layer = nn.Linear(self.embedding_dim, 50)

    def forward(self, sentence):
        embedding_out = self.embedding(sentence)
        # BoW layer to generate sentence embedding from word embedding
        bow_out = torch.mean(embedding_out, dim=1)
        fully_connected_out = self.fully_connected_layer(bow_out)
        return fully_connected_out


class BiLSTMClassifier(nn.ModuleList):
    def __init__(self, config, num_words, embedding_matrix=None, device=None):
        super().__init__()
        self.device = device
        self.embedding_dim = int(config['PARAMETER']['embedding_dim'])
        self.lstm_layers = int(config['PARAMETER']['lstm_layers'])
        self.lstm_hidden_dim = int(config['PARAMETER']['lstm_hidden_dim'])

        # embedding layer
        if embedding_matrix is not None:
            # use pre-trained embedding
            if config['PARAMETER']['freeze'] == 'True':
                # freeze the embedding layer and the parameters will not update during training
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
            else:
                # fine-tune parameters of embedding layer during training
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            # randomly initialize the embedding layer (the parameters will update during training)
            self.embedding = nn.Embedding(num_words, self.embedding_dim)

        # BiLSTM layer to generate sentence embedding from word embedding
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        # classification layer to generate the final classification
        self.fully_connected_layer = nn.Linear(in_features=self.lstm_hidden_dim * 2, out_features=50)

    def forward(self, x):
        # get the output of embedding layer
        embedding_out = self.embedding(x)
        # initialize the hidden state and cell state
        h = torch.zeros((self.lstm_layers * 2, x.size(0), self.lstm_hidden_dim)).to(self.device)
        c = torch.zeros((self.lstm_layers * 2, x.size(0), self.lstm_hidden_dim)).to(self.device)
        # get the output of lstm layer
        lstm_out, (hidden, cell) = self.lstm(embedding_out, (h, c))
        lstm_out = lstm_out[:, -1, :]  # get the hidden state of the last time step
        # get the final output of classification layer
        fully_connected_out = self.fully_connected_layer(lstm_out)

        return fully_connected_out


class BowBiLSTMClassifier(nn.ModuleList):
    def __init__(self, config, num_words, embedding_matrix=None, device=None):
        super().__init__()
        self.device = device
        # same structure with BiLSTM model except for the input neurons of fully connected layer
        self.embedding_dim = int(config['PARAMETER']['embedding_dim'])
        self.lstm_layers = int(config['PARAMETER']['lstm_layers'])
        self.lstm_hidden_dim = int(config['PARAMETER']['lstm_hidden_dim'])

        if embedding_matrix is not None:
            if config['PARAMETER']['freeze'] == 'True':
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
            else:
                self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(num_words, self.embedding_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        # as it combines the bow and bilstm, the input features need to be increased
        self.fully_connected_layer = nn.Linear(in_features=self.lstm_hidden_dim * 2 + self.embedding_dim,
                                               out_features=50)

    def forward(self, x):
        h = torch.zeros((self.lstm_layers * 2, x.size(0), self.lstm_hidden_dim)).to(self.device)
        c = torch.zeros((self.lstm_layers * 2, x.size(0), self.lstm_hidden_dim)).to(self.device)

        embedding_out = self.embedding(x)

        bow_out = torch.mean(embedding_out, dim=1)
        lstm_out, (hidden, cell) = self.lstm(embedding_out, (h, c))
        lstm_out = lstm_out[:, -1, :]

        # combine the bow sentence vectors and bilstm sentence vectors by horizontal concatenation
        combined_out = torch.cat((bow_out, lstm_out), 1)
        fully_connected_out = self.fully_connected_layer(combined_out)

        return fully_connected_out
