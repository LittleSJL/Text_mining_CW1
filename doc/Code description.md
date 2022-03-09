# Structure
The whole system is divided into four modules with five code files
- `Main code file`: question_classifier.py
- `Input module`: input_handler.py
- `Model module`: modeling.py, model_handler.py
- `Output module`: output_handler.py

# Main code file
## question_classifier.py
#### Function
None
#### Class
None
#### Description
This module has no function and class. It just uses classes and functions implemented in other modules to build a standard machine learning workflow to deal with the question classification task: 
- `Input module`: read configuration file; create a input handler to prepare the input for the model training and testing.
- `Model module`: in training process: build, train and save the model; in testing process, load the model and get the predictions. 
- `Output module`: create a output handler to do result evaluation and write results into the file.
# Input module
## Input_handler.py
#### Function outside the class
- `print_info()`: print different configuration settings information at the beginning.
- `load_config_file(config_path)`: load the configuration file which stores all needed information.
- `load_stopwords(stopwords_path)`: load the stopwords list used in pre-processing steps
- `load_mapping(mapping_path)`: load char_to_id file or label_to_id file
	- `char_to_id/label_to_id`: dictionaries which map a word/label to a unique id
- `load_embedding_dic(embedding_path)`: load pre-trained embedding dictionary(glove)
- `if_contain_non_alphabet(word)`: check if a word contains non-alphabet characters
- `remove_unrelated_words(text, stopwords)`: remove stopwords and words which contain non-alphabet characters
- `pre_process(raw_text, stopwords)`: input a raw sentence and output a clean sentence by:
	- lowercase
	- remove punctuations
	- remove unrelated words
- `load_data(data_path, stopwords)`: read the file and pre-process the data
- `convert_sentence_to_vectors(char_to_id, sentence, max_length)`: input a text sentence and output a fixed-length(max_length) tensor vectors by mapping each token to its unique id
- `convert_label_to_one_hot_encoding(label_to_id, label)`: input a text label and output a  one-hot encoding tensor 
- `build_data_loader(sentence_list, label_list, batch_size)`: build data loader for batch training
#### Class
`InputHandler`
#### Function inside the class
- `init(self, config)`: set up the intput handler such as storing the config, loading stopwords, char_to_id and label_to_id file.
- `get_training_data(self)`: load and pre-process the training/dev data
- `get_test_data(self)`: load and pre-process the test data
- `get_char_to_id(self)`: return char_to_id dictionary
- `transform_input(self, sentences, labels)`: transform the data and prepare the input for model training and test
- `get_embedding_matrix(self)`: load pre-trained embedding file and create embedding matrix (used to initialize the embedding layer in neural network)
#### Description
The main functions of this module are as follows:
- Load and pre-process the raw data.
- Load other files, such as char_to_id/label_to_id file, embedding file.
- Transform and prepare the data that can be used in the model training and testing. For example, the data should be fixed-length tensors and label should be a one-hot encoding tensors.
# Model module
## modeling.py
#### Class
- `BowClassifier`
- `BiLSTMClassifier`
- `BowBiLSTMClassifier`
#### Function of Class BowClassifier: 
- `init(self, config, num_words, embedding_matrix)`: define the network structure of classifier based on BoW(bag of words) method
- `forward(self, sentence)`: define the forward computation of the network based on BoW

#### Function of Class BiLSTMClassifier: 
- `init(self, config, num_words, embedding_matrix)`: define the network structure of classifier based on BiLSTMClassifier method
- `forward(self, sentence)`: define the forward computation of the network based on BiLSTM

#### Function of Class BowBiLSTMClassifier: 
- `init(self, config, num_words, embedding_matrix)`: define the network structure of classifier based on BowBiLSTM method: 
	- combine the sentence vectors from BoW and BiLSTM through horizontal concatenation then send the new vector into the following classification layer
- `forward(self, sentence)`: define the forward computation of the network based on BowBiLSTM
#### Description
The main function of this code file in model module is as follows: define three different classifier models: 
- define their structures
- define their forward computation steps

## model_handler.py
#### Class
`ModelHandler`
#### Function inside the class
- `init(self, config, num_words, embedding_matrix, model_path)`: set up the model handler such as storing the configuration object and build the model
- `build_model(self, num_words, embedding_matrix, model_path)`: 
	- build the model based on different selections
	- if model_path is specified, then load the model.
- `train(self, x_train, y_train, x_dev, y_dev)`: 
	- preparation: build data loader, specify loss criteria and optimizer
	- start training: forward to get the prediction, backward to update the parameters
	- early stopping: calculate the accuracy on development set during each epoch, stop training if the best accuracy is not updated for a number of epochs (pre-defined)
	- save model: save the model that achieves the best accuracy on dev set
- `predict(self, x_test)`: take a question as input and output the classification result through the forward computation of the model
- `save_model(self)`: save the best model in training process
- `load_model(self, path)`: load pre-trained model in training process
#### Description
The main function of this code file in model module is as follows: define the training, predicting, saving and loading steps of the model.  Especially for the training process: 
- `Best accuracy`: define a variable to store the best accuracy on dev set during training. Update it after each epoch
- `Early stopping`: stop training if the best accuracy is not updated for a number of epochs (pre-defined)
- `Best model`: save the best model (best accuracy on dev set) after the training.
# Output module
## output_handler.py
#### Function outside the class
- `transform_output(one_hot_predict, one_hot_true)`: transform the one-hot encoding tensor to numpy int label
- `get_id_to_label(path)`: create a id_to_label dictionary that maps a unique id to its label
- `build_classes(int_predict, int_true, id_to_label)`: get the text classes used for plotting confusion matrix and displaying f1 score of each class
- `evaluate_during_training(one_hot_predict, one_hot_true)`: used for calculating the accuracy and f1 score during training
- `plot_confusion_matrix_figure(true, predict, classes, save_img_path)`: plot, display and save the confusion matrix
#### Class
`OutHandler`
#### Function inside the class
- `init__(self, config, raw_sentence)`: set up the output handler such as storing the config and initializing variables
- `write_result(self)`: write the evaluation results into files
- `result_evaluation(self, one_hot_predict, one_hot_true)`: evaluate the model based on the prediction and true label
#### Description
The main functions of this module are as follows:
- `Prepare for evaluation`: Transform the output (the raw outputs are one-hot encoding tensors) from the model and prepare for evaluation
- `Result evaluation`: calculate the accuracy, three types of F1 score (micro, macro, weighted), F1 score for each class, confusion matrix
- `Write the results`: 
	- accuracy and total three types of f1 score 
	- f1 score of each class 
	- the original sentence, prediction label and true label for each testing question


