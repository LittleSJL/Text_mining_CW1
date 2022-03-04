from src.model_handler import ModelHandler
from src.output_handler import result_evaluation
from sklearn.model_selection import train_test_split
from src.input_handler import read_config_file, load_data, load_embeddings, transform_input, load_mapping


"""
Input module: 
(1) read config file (2) load data (3) char_to_id file (4) transform input (5) load embedding (if needed)
"""
# read configuration file
config_path = "../data/bilstm_random_finetune.ini"
config = read_config_file(config_path)

# load and pre-process the training data and test data
_, x_train, y_train = load_data(config['PATH']['training_path'], config['PATH']['stopwords_path'])
raw_x_test, x_test, y_test = load_data(config['PATH']['testing_path'], config['PATH']['stopwords_path'])
# split training data into 90% for training, 10% for development
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=2)

# load char_to_id dictionary
char_to_id = load_mapping(path=config['PATH']['char_to_id_path'])

# transform text data and label into tensor vectors
train_data, train_label = transform_input(sentences=x_train, labels=y_train, char_to_id=char_to_id, config=config)
test_data, test_label = transform_input(sentences=x_test, labels=y_test, char_to_id=char_to_id, config=config)
dev_data, dev_label = transform_input(sentences=x_dev, labels=y_dev, char_to_id=char_to_id, config=config)


# firstly set is_training = True: run the code to train and save the model
# then set is_training = False: run the code to load, test the model and write the evaluation results
is_training = False
embedding_matrix = None
if config['PARAMETER']['embedding_selection'] == 'Glove' and is_training:
    # load pre-trained embedding
    embedding_matrix = load_embeddings(config['PATH']['embedding_path'],
                                       config['PARAMETER']['embedding_dim'], char_to_id)

"""
Model module: 
(1) training: build, train and save the model (2) test: load and test the model

Output module:
(1) transform output (2) calculate the accuracy and f1 score (3) write the evaluation results
"""
if is_training:
    # build, train and save the model
    model = ModelHandler(config=config, embedding_matrix=embedding_matrix, char_to_id=char_to_id)
    model.train(train_data, train_label, dev_data, dev_label)
else:
    # build, load and test the model
    model = ModelHandler(config=config, char_to_id=char_to_id, model_path=config['PATH']['model_path'])
    prediction = model.test(test_data)
    # do evaluation and write the results into .txt file
    result_evaluation(config, raw_x_test, test_label, prediction)
