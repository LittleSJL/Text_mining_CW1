import os
import torch
import random
from src.input_handler import build_data_loader
from src.modeling import BiLSTMClassifier, BowClassifier, BowBiLSTMClassifier
from src.output_handler import calculate_evaluation_results

random.seed(1)
torch.manual_seed(1)


class ModelHandler:
    def __init__(self, config, num_words, embedding_matrix=None, model_path=None):
        self.config = config
        self.model = None
        # used to store the model weights that achieve the best accuracy on development during training
        self.best_model_state = None

        # build BiLSTM model
        if self.config['PARAMETER']['model_selection'] == 'BiLSTM':
            self.model = BiLSTMClassifier(config=self.config, num_words=num_words,
                                          embedding_matrix=embedding_matrix)
        # build Bag of Word (BoW) model
        elif self.config['PARAMETER']['model_selection'] == 'BoW':
            self.model = BowClassifier(config=self.config, num_words=num_words, embedding_matrix=embedding_matrix)
        # build BoW_BiLSTM model
        elif self.config['PARAMETER']['model_selection'] == 'BoWBiLSTM':
            self.model = BowBiLSTMClassifier(config=self.config, num_words=num_words, embedding_matrix=embedding_matrix)
        # if model_path is specified, load the model
        if model_path:
            self.load_model(model_path)

    def train(self, x_train, y_train, x_dev, y_dev):
        # build data loader for training
        loader_training = build_data_loader(x_train, y_train, batch_size=int(self.config['PARAMETER']['batch_size']))
        # specify loss criteria and optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config['PARAMETER']['learning_rate']))
        # store the best accuracy on development set, which will be use for early stopping
        best_dev_accuracy = 0
        early_stopping_counter = 0

        print('-------------------Start training-------------------')
        for epoch in range(int(self.config['PARAMETER']['epochs'])):
            self.model.train()
            loss_list = []
            accuracy_list = []
            for x_batch, y_batch in loader_training:
                # forward to get the prediction
                y_pred = self.model(x_batch.to(torch.int))
                # calculate loss and backward to update the parameters
                loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # calculate and store the accuracy as well as loss during training
                accuracy_training, _ = calculate_evaluation_results(y_pred, y_batch)
                accuracy_list.append(accuracy_training)
                loss_list.append(loss.detach().numpy())
            # loss and accuracy on training set
            loss_train = sum(loss_list) / len(loss_list)
            accuracy_train = sum(accuracy_list) / len(accuracy_list)
            # test the model on development set
            pred_dev = self.test(x_dev)
            accuracy_dev, _ = calculate_evaluation_results(pred_dev, y_dev)

            # update training states for accuracy and early stopping
            if accuracy_dev > best_dev_accuracy:
                """
                if the current accuracy is better than the best accuracy before,
                then clear early stopping counter and update the model weights (always store the weights of best model)
                """
                best_dev_accuracy = accuracy_dev
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
        print("Best accuracy on development set:", best_dev_accuracy)
        self.save_model()

    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_test)
            y_pred = torch.softmax(y_pred, dim=1)
        return y_pred

    def save_model(self):
        print('Saving the best model to', self.config['PATH']['model_path'], '\n')
        save_model_path = os.path.join(self.config['PATH']['model_path'])
        torch.save(self.best_model_state, save_model_path)

    def load_model(self, path):
        print('Loading model from', path, '\n')
        self.model.load_state_dict(torch.load(path))
