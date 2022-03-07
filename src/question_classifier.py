from model_handler import ModelHandler
from output_handler import OutputHandler
from input_handler import load_config_file, InputHandler


# read configuration file
config_path = "../data/bow_glove_finetune.ini"
config = load_config_file(config_path)
# build input_handler
input_handler = InputHandler(config)

"""
Firstly set is_training = True: run the code to train and save the model.
Then set is_training = False: run the code to load, test the model and write the evaluation results
"""
is_training = False
if is_training:
    # load and pre-process the training data and dev data for training
    x_train, x_dev, y_train, y_dev = input_handler.get_training_data()
    # transform text data and label into tensor vectors
    train_data, train_label = input_handler.transform_input(sentences=x_train, labels=y_train)
    dev_data, dev_label = input_handler.transform_input(sentences=x_dev, labels=y_dev)
    # build, train and save the model
    model = ModelHandler(config=config, num_words=len(input_handler.get_char_to_id()),
                         embedding_matrix=input_handler.get_embedding_matrix())
    model.train(train_data, train_label, dev_data, dev_label)
else:
    # load and pre-process the training data and dev data for training
    x_train, x_dev, y_train, y_dev = input_handler.get_training_data()
    # transform text data and label into tensor vectors
    train_data, train_label = input_handler.transform_input(sentences=x_train, labels=y_train)
    dev_data, dev_label = input_handler.transform_input(sentences=x_dev, labels=y_dev)

    # load and pre-process the test data for testing
    raw_x_test, x_test, y_test = input_handler.get_test_data()
    # transform text data and label into tensor vectors
    test_data, test_label = input_handler.transform_input(sentences=x_test, labels=y_test)
    # build, load and test the model
    model = ModelHandler(config=config, num_words=len(input_handler.get_char_to_id()),
                         model_path=config['PATH']['model_path'])
    prediction = model.predict(test_data)
    # build output_handler
    output_handler = OutputHandler(config, raw_x_test)
    # do evaluation
    output_handler.result_evaluation(prediction, test_label)
    # write evaluation results
    output_handler.write_result()
