# Folder structure
There are three folders:
- `Document`: 
	- `code description.md`: descriptions for the functions in each code file in src folder;
	- `README.md` (this file): instructions on how to run the code
- `Data`: 
	- `training_set.txt, test_set.txt`: training and test data set (dev set is generated during training);
	- `9 configuration files` (e.g., `bow_glove_freeze.ini`);
	- `stopwords.txt`: stores the stopwords; 
	- `char_to_id.pkl, label_to_id.pkl`: file that map words/labels to their unique id;
- `Src`: the source code (the `code description.md` will explain the its structure and details)


# Instructions on how to use the code
There are two phases that you can choose: train and test.
- To train a model, `run python question_classifier.py train -config [configuration_file_path]`
	- `During training`: the system will print the training loss, training accuracy, dev accuracy and early stopping counter of each epoch.
	- `After training`: it will save the model that achieves the best accuracy on dev set to the path specified in the configuration file.
- To test the model, `run python question_classifier.py test -config [configuration_file_path]`
	- `During testing`: the system will print the accuracy and three types of F1 score on test set. 
	- `After testing`: the following results will be saved to the data folder:
		- `result.txt`: 
			- accuracy on test set
			- f1 score on test set (micro, macro, weighted); f1 scores of each class; 
			- the original sentence, prediction label and true label for each testing question.
		- `Confusion martix image`: a image that displays a 50*50 matrix

**Reminder**: Because there is no trained model in the data folder, you need to first train and save a model. If you run the test command without training first, the system will randomly initialize a new model and get poor evaluation results on the test set.

# Configuration files
You can choose different model settings by modifying the [configuration_file_path] in the command line.
- `bow_random_finetune.ini`: BoW model, randomly initialize the embedding, finetune the embedding during training.
- `bow_glove_freeze.ini`: BoW model, use Glove embedding, freeze the embedding during training.
- `bow_glove_finetune.ini`: BoW model, use Glove embedding, finetune the embedding during training.
- `bilstm_random_finetune.ini`: BiLSTM model, randomly initialize the embedding, finetune the embedding during training.
- `bilstm_glove_freeze.ini`: BiLSTM model, Glove embedding, freeze the embedding during training.
- `bilstm_glove_finetune.ini`: BiLSTM model, Glove embedding, finetune the embedding during training.
 -`bowbilstm_random_finetune.ini`: BoWBiLSTM model, randomly initialize the embedding, finetune the embedding during training.
- `bowbilstm_glove_freeze.ini`:BoWBiLSTM model, Glove embedding, freeze the embedding during training.
 -`bowbilstm_glove_finetune.ini`: BoWBiLSTM model, Glove embedding, finetune the embedding during training.

# Fields
There are two sections in a configuration file: `PATH` section and `PARAMETER` section
- In the `PATH` section, all the file paths are specified:
	- `training_path, test_path`: the path to training and test data
	- `stopwords_path`: the path to stopwords file
	- `embedding_path`: the path to pre-trained embedding file
	- `char_to_id_path, label_to_id_path`: the path to char mapping, label mapping file
	- `model_path`: the path to save and load models
	- `result_path`: the path to save evaluation result
	- `confusion_martix_path`: the path to save confusion matrix image
- In the `PARAMETER` section, all the hyperparameters are specified:
	- `embedding_selection`: specify which embeddings to use, [Glove, Random initialize]
	- `freeze`: whether to freeze the embedding weights during training, [True, False]
	- `model_selection`: specify which models to use, [BoW, BiLSTM, BoWBiLSTM]
	- `embedding_dim`: the dimension of word embedding
	- `lstm_layers`: the number of layers in BiLSTM or BoWBiLSTM models
	- `lstm_hidden_dim`: the dimension of hidden layers in BiLSTM or BoWBiLSTM models
	- `epochs`: the number of times that the model will train on the training set
	- `max_len`: the max sequence length in data preprocessing
	- `batch_size`: the number of training examples used in one iteration
	- `early_stopping`: the early stopping threshold to stopping training
	- `learning_rate`: the value of learning rate












