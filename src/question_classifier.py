from src.model_handler import ModelHandler
from src.output_handler import result_evaluation
from src.input_handler import read_config_file, InputHandler


# read configuration file
config_path = "../data/bilstm_random_finetune.ini"
config = read_config_file(config_path)
# build preprocessor
preprocessor = InputHandler(config)

# firstly set is_training = True: run the code to train and save the model
# then set is_training = False: run the code to load, test the model and write the evaluation results
is_training = False
if is_training:
    # load and pre-process the training data and dev data for training
    x_train, x_dev, y_train, y_dev = preprocessor.get_training_data()
    # transform text data and label into tensor vectors
    train_data, train_label = preprocessor.transform_input(sentences=x_train, labels=y_train)
    dev_data, dev_label = preprocessor.transform_input(sentences=x_dev, labels=y_dev)
    # build, train and save the model
    model = ModelHandler(config=config, num_words=len(preprocessor.get_char_to_id()),
                         embedding_matrix=preprocessor.get_embedding_matrix())
    model.train(train_data, train_label, dev_data, dev_label)
else:
    # load and pre-process the test data for testing
    raw_x_test, x_test, y_test = preprocessor.get_test_data()
    # transform text data and label into tensor vectors
    test_data, test_label = preprocessor.transform_input(sentences=x_test, labels=y_test)
    # build, load and test the model
    model = ModelHandler(config=config, num_words=len(preprocessor.get_char_to_id()),
                         model_path=config['PATH']['model_path'])
    prediction = model.test(test_data)
    # do evaluation and write the results into .txt file
    result_evaluation(config, raw_x_test, test_label, prediction)
