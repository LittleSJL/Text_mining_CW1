# Structure
The whole system consists of four modules with five code files
- `Main code file`: question_classifier.py
- `Input module`: input_handler.py
- `Model module`: modeling.py, model_handler.py
- `Output module`: output_handler.py

# Main code file
## question_classifier.py
#### Description
This module has no function and class. It firstly uses `argparse` to parse command line arguments. Then it uses classes and functions implemented in other modules to build a standard machine learning workflow to deal with the question classification task: 
- `Input module`: create a input handler to prepare the input for the model training and testing.
- `Model module`: in training process: build, train and save the model; in testing process, load the model and get the predictions. 
- `Output module`: create a output handler to do result evaluation and write results into the file.
#### Class
None
#### Function
None

# Input module
## Input_handler.py
#### Description
The main function of this module is to load, pre-process and transform the raw data to prepare for model training and testing.
#### Function outside the class
- `load_config_file(config_path)`: load the configuration file which stores all needed information.
- `print_info()`: print different configuration settings information at the beginning.
- `load_stopwords(stopwords_path)`: load the stopwords list used in pre-processing steps.
- `load_mapping(mapping_path)`: load char_to_id file or label_to_id file 
	- `char_to_id/label_to_id`: dictionaries which map a word/label to a unique id.
- `load_embedding_dic(embedding_path)`: load pre-trained embedding dictionary(glove).
- `if_contain_non_alphabet(word)`: check if a word contains non-alphabet characters,
- `remove_unrelated_words(text, stopwords)`: remove stopwords and words which contain non-alphabet characters.
- `pre_process(raw_text, stopwords)`: input a raw sentence and output a clean sentence by lowercase, remove punctuations, remove unrelated words.
- `load_data(data_path, stopwords)`: load and pre-process the data.
    - input: path of data file and list of stopwords.
    - algorithm: read the file and pre-process the data.
    - output: raw sentences, clean sentences and the labels for each sentence.
- `convert_sentence_to_vectors(char_to_id, sentence, max_length)`: input a text sentence and output a fixed-length(max_length) tensor vectors by mapping each token to its unique id.
- `convert_label_to_one_hot_encoding(label_to_id, label)`: input a text label and output a  one-hot encoding tensor.
- `build_data_loader(sentence_list, label_list, batch_size)`: build data loader for batch training.
#### Class
`InputHandler`
#### Function inside the class
- `init(self, config)`: set up the input handler such as storing the config, loading stopwords, char_to_id and label_to_id file.
- `get_training_data(self)`: load, pre-process the training data and split it into 90% for training and 10% for validation(dev set).
- `get_test_data(self)`: load and pre-process the test data.
- `get_char_to_id(self)`: return char_to_id dictionary.
- `transform_input(self, sentences, labels)`: transform the input and prepare for model training and test.
    - input: clean sentences, labels.
    - algorithmL: convert the sentences/labels to fixed-length tensors/one-hot encoding tensors.
    - output: fixed-length tensors/one-hot encoding tensors.
- `get_embedding_matrix(self)`: load pre-trained embedding file and create embedding matrix (used to initialize the embedding layer in neural network).

# Model module
## modeling.py
#### Description
The main function of this code file in model module is to define three different classifier models (structures and forward computation steps).
#### Class
- `BowClassifier`
- `BiLSTMClassifier`
- `BowBiLSTMClassifier`
#### Function of Class BowClassifier: 
- `init(self, config, num_words, embedding_matrix)`: define the network structure of classifier based on BoW(bag of words) method.
- `forward(self, sentence)`: define the forward computation of the network based on BoW.

#### Function of Class BiLSTMClassifier: 
- `init(self, config, num_words, embedding_matrix)`: define the network structure of classifier based on BiLSTMClassifier method.
- `forward(self, sentence)`: define the forward computation of the network based on BiLSTM.

#### Function of Class BowBiLSTMClassifier: 
- `init(self, config, num_words, embedding_matrix)`: define the network structure of classifier based on BowBiLSTM method: combine the sentence vectors from BoW and BiLSTM through horizontal concatenation to create a new sentence vector.
- `forward(self, sentence)`: define the forward computation of the network based on BowBiLSTM.

## model_handler.py
#### Description
The main function of this code file in model module is to define the training, predicting, saving and loading steps of the model.
#### Class
`ModelHandler`
#### Function inside the class
- `init(self, config, num_words, embedding_matrix, model_path)`: set up the model handler such as storing the configuration object and build the model.
- `build_model(self, num_words, embedding_matrix, model_path)`: build the model based on configuration settings. If model_path is specified, then load the model.
- `train(self, x_train, y_train, x_dev, y_dev)`: define the training process of the model.
	- preparation: build data loader, specify loss criteria and optimizer.
	- start training: forward to get the prediction, backward to update the parameters.
	- early stopping: calculate the accuracy on development set after each epoch, stop training if the best accuracy is not updated for a number of epochs (pre-defined).
	- save model: save the model that achieves the best accuracy on dev set.
- `predict(self, x_test)`: take questions as input and output the classification results through the forward computation of the model.
- `save_model(self)`: save the best model in training process.
- `load_model(self, path)`: load pre-trained model in training process.

# Output module
## output_handler.py
#### Description
The main functions of this module are as follows:
- `Prepare for evaluation`: Transform the raw model predictions and prepare for evaluation.
- `Result evaluation`: evaluate the model based on the predictions and true labels.
- `Write the results`: write the evaluation results into files.
#### Function outside the class
- `transform_output(one_hot_predict, one_hot_true)`: transform the raw model output
    - input: raw model predictions, raw true labels (one-hot encoding tensor)
    - algorithm: convert the one-hot encoding tensor to numpy int labels
    - output: numpy int predictions and true labels.
- `get_id_to_label(path)`: create a id_to_label dictionary that maps a unique id to its text label.
- `build_classes(int_predict, int_true, id_to_label)`: get the text classes used for plotting confusion matrix and displaying f1 score of each class.
- `evaluate_during_training(one_hot_predict, one_hot_true)`: used for calculating the accuracy and f1 score during training.
- `plot_confusion_matrix_figure(true, predict, classes, save_img_path)`: plot and save the confusion matrix image.
#### Class
`OutHandler`
#### Function inside the class
- `init__(self, config, raw_sentence)`: set up the output handler such as storing the config and initializing variables.
- `result_evaluation(self, one_hot_predict, one_hot_true)`: evaluate the model based on the predictions and true labels.
    - calculate the accuracy, three types of F1 score (micro, macro, weighted).
    - calculate F1 score for each class.
    - calculate the confusion matrix and save the image.
- `write_result(self)`: write the evaluation results into files.
	- accuracy and three types of F1 score (micro, macro, weighted).
	- f1 score of each class in ascending order.
	- the original sentences, prediction labels and true labels for each testing question.



