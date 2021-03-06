import os
import torch
import random
from input_handler import build_data_loader
from output_handler import evaluate_during_training
from modeling import BiLSTMClassifier, BowClassifier, BowBiLSTMClassifier

random.seed(1)
torch.manual_seed(1)


class ModelHandler:
    def __init__(self, config, num_words, embedding_matrix=None, model_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        # used to store the model weights that achieve the best accuracy on development during training
        self.best_model_state = None
        self.build_model(num_words, embedding_matrix, model_path)

    def build_model(self, num_words, embedding_matrix, model_path):
        # build BiLSTM model
        if self.config['PARAMETER']['model_selection'] == 'BiLSTM':
            self.model = BiLSTMClassifier(config=self.config, num_words=num_words,
                                          embedding_matrix=embedding_matrix, device=self.device)
        # build Bag of Word (BoW) model
        elif self.config['PARAMETER']['model_selection'] == 'BoW':
            self.model = BowClassifier(config=self.config, num_words=num_words,
                                       embedding_matrix=embedding_matrix, device=self.device)
        # build BoW_BiLSTM model
        elif self.config['PARAMETER']['model_selection'] == 'BoWBiLSTM':
            self.model = BowBiLSTMClassifier(config=self.config, num_words=num_words,
                                             embedding_matrix=embedding_matrix, device=self.device)
        # if model_path is specified, load the model
        if model_path:
            self.load_model(model_path)
        self.model.to(self.device)

    def train(self, x_train, y_train, x_dev, y_dev):
        # build data loader for training
        loader_training = build_data_loader(x_train, y_train, batch_size=int(self.config['PARAMETER']['batch_size']))
        # specify loss criteria and optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config['PARAMETER']['learning_rate']))
        # store the best accuracy on development set, which will be use for early stopping
        best_dev_acc = 0
        early_stopping_counter = 0

        print('-------------------Start training-------------------')
        for epoch in range(int(self.config['PARAMETER']['epochs'])):
            self.model.train()
            loss_list = []
            accuracy_list = []
            for x_batch, y_batch in loader_training:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # forward to get the prediction
                y_pred = self.model(x_batch.to(torch.int))
                # calculate loss and backward to update the parameters
                loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # calculate and store the accuracy as well as loss during training
                accuracy_training, f1_training = evaluate_during_training(y_pred, y_batch)
                accuracy_list.append(accuracy_training)
                loss_list.append(loss.cpu().detach().numpy())
            # loss and accuracy on training set
            loss_train = sum(loss_list) / len(loss_list)
            accuracy_train = sum(accuracy_list) / len(accuracy_list)
            # test the model on development set
            x_dev = x_dev.to(self.device)
            y_dev = y_dev.to(self.device)
            pred_dev = self.predict(x_dev)
            accuracy_dev, f1_dev = evaluate_during_training(pred_dev, y_dev)

            # update training states for accuracy and early stopping
            if accuracy_dev > best_dev_acc:
                """
                if the current accuracy is better than the best accuracy before,
                then clear early stopping counter and update the model weights (always store the weights of best model)
                """
                best_dev_acc = accuracy_dev
                early_stopping_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                # else, increase the early_stopping_counter
                early_stopping_counter += 1
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Dev accuracy: %.5f, Early stopping: %d" %
                  (epoch + 1, loss_train, accuracy_train, accuracy_dev, early_stopping_counter))
            if early_stopping_counter >= int(self.config['PARAMETER']['early_stopping']):
                # stop training if the best accuracy is not updated for a number of epochs (pre-defined)
                print('Early stop training...')
                break
        print('-------------------End training-------------------')
        print("Best accuracy on development set:", best_dev_acc)
        self.save_model()

    def predict(self, x_test):
        # take a question as input and output the classification result (one-hot encoding tensor)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_test.to(self.device))
            y_pred = torch.softmax(y_pred, dim=1)
        return y_pred

    def save_model(self):
        # save the best model in training process
        print('Saving the best model to', self.config['PATH']['model_path'], '\n')
        save_model_path = os.path.join(self.config['PATH']['model_path'])
        torch.save(self.best_model_state, save_model_path)

    def load_model(self, path):
        try:
            # load pre-trained model in training process
            print('Loading model from', path, '\n')
            self.model.load_state_dict(torch.load(path))
        except OSError as e:
            print("Warning: No model to load and just randomly initialize a model to test."
                  " If you want to test a pre-trained model, please run the code to train and save a model first\n")
