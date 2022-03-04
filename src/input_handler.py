import re
import torch
import random
import pickle
import numpy as np
import configparser
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

random.seed(1)
torch.manual_seed(1)


def print_info(config):
    """
    print different config settings
    """
    print('Embedding selection:', config['PARAMETER']['embedding_selection'])
    if config['PARAMETER']['freeze'] == 'True':
        print('Freeze or fine-tune: Freeze')
    else:
        print('Freeze or fine-tune: Fine-tune')
    print('Model selection:', config['PARAMETER']['model_selection'])


def read_config_file(config_path):
    """
    Read configuration file:
        Embedding selection: [Glove, Random initialize]
        Freeze or Fine-tune: [Freeze, Fine-tune]
        Model selection: [BiLSTM, BoW]
    """
    config = configparser.ConfigParser()
    config.read(config_path)  # 6 different configuration files
    config.sections()
    print_info(config)
    return config


def load_stopwords(stopwords_path):
    """
    load stopwords file
    """
    stopwords_file = open(stopwords_path, 'r')
    stopwords_list = stopwords_file.readlines()
    stopwords_file.close()
    stopwords_list = [word.strip('\n') for word in stopwords_list]
    return stopwords_list


def load_mapping(path):
    """
    load char_to_id file or label_to_id file
    """
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    return mapping


def load_embedding_dic(embedding_path):
    """
    load pre-trained embedding dictionary(glove)
    """
    embedding_dic = {}
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embedding_dic[word] = vec
    return embedding_dic


def if_contain_non_alphabet(word):
    """
    check if a word contains non-alphabet characters
    """
    for character in word:
        if not character.isalpha():
            return True
    return False


def remove_unrelated_words(text, stopwords):
    """
    remove stopwords and words which contain non-alphabet characters
    """
    sentence_list = []
    for word in text.split():
        if word not in stopwords and not if_contain_non_alphabet(word):
            sentence_list.append(word)
    remove_unrelated_words_text = " ".join(sentence_list)
    return remove_unrelated_words_text


def pre_process(raw_text, stopwords):
    """
    (1)lowercase (2)remove punctuation (3)remove unrelated words
    """
    # lowercase all words in the sentence
    lower_case_text = raw_text.lower()
    # remove punctuations
    punctuation = "[`!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"
    remove_punc_text = re.sub(punctuation, ' ', lower_case_text)
    remove_punc_text = ' '.join(remove_punc_text.split())
    # remove unrelated words
    clean_text = remove_unrelated_words(remove_punc_text, stopwords)

    return clean_text


def load_data(data_path, stopwords):
    """
    read the file and pre-process the data
    """
    # read the .txt file
    data_file = open(data_path, 'r', encoding="ISO-8859-1")
    raw_data = data_file.readlines()
    data_file.close()

    # pre-process the raw data
    raw_question = []
    clean_question = []
    label = []
    for raw_data_item in raw_data:
        # split a line of string into a question and its label
        split_index = raw_data_item.index(' ')
        label_item = raw_data_item[:split_index]
        raw_question_item = raw_data_item[split_index:]
        # pre-process the data
        clean_question_item = pre_process(raw_question_item, stopwords)
        # store each clean question and the label
        raw_question.append(raw_question_item)
        clean_question.append(clean_question_item)
        label.append(label_item)

    return raw_question, clean_question, label


def convert_sentence_to_vectors(char_to_id, sentence, max_length):
    """
    convert a text sentence to a fixed-length(max_length) vectors by mapping each token to its unique id
    """
    sentence_vector = torch.zeros(max_length, dtype=torch.int)
    counter = -1
    for token in sentence.split()[:max_length]:
        counter += 1
        if token in char_to_id:
            sentence_vector[counter] = char_to_id[token]
            continue
        # if a token is not in the vocabulary, then assign it as #UNK#
        sentence_vector[counter] = char_to_id["#UNK#"]
    return sentence_vector


def convert_label_to_one_hot_encoding(label_to_id, label):
    """
    convert a text label to one-hot encoding
    """
    label_vector = torch.zeros(len(label_to_id))
    label_vector[label_to_id.get(label) - 1] = 1
    return label_vector


def build_data_loader(sentence_list, label_list, batch_size=1):
    """
    concatenate the sentence and label to build data loader
    """
    concat_sentence_label = []
    for i in range(len(sentence_list)):
        concat_sentence_label.append([sentence_list[i], label_list[i]])
    data_loader = DataLoader(concat_sentence_label, batch_size=batch_size, shuffle=False)
    return data_loader


class InputHandler:
    def __init__(self, config):
        self.config = config
        self.training_path = self.config['PATH']['training_path']
        self.test_path = self.config['PATH']['test_path']
        self.stopwords = load_stopwords(self.config['PATH']['stopwords_path'])
        self.char_to_id = load_mapping(self.config['PATH']['char_to_id_path'])
        self.label_to_id = load_mapping(self.config['PATH']['label_to_id_path'])

    def get_training_data(self):
        # load and pre-process training data
        _, x_train, y_train = load_data(self.training_path, self.stopwords)
        # split training data into 90% for training, 10% for development
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=2)
        return x_train, x_dev, y_train, y_dev

    def get_test_data(self):
        # load and pre-process test data
        raw_x_test, x_test, y_test = load_data(self.test_path, self.stopwords)
        return raw_x_test, x_test, y_test

    def get_char_to_id(self):
        return self.char_to_id

    def transform_input(self, sentences, labels):
        # transform sentences from text to vectors
        sentence_vectors = torch.zeros(len(sentences), int(self.config['PARAMETER']['max_len']), dtype=torch.int)
        counter = 0
        for sentence in sentences:
            sentence_vectors[counter] = convert_sentence_to_vectors(self.char_to_id, sentence,
                                                                    int(self.config['PARAMETER']['max_len']))
            counter += 1

        # transform labels from text to one-hot encoding
        one_hot_encoding_label = torch.zeros(len(labels), len(self.label_to_id))
        counter = 0
        for label in labels:
            one_hot_encoding_label[counter] = convert_label_to_one_hot_encoding(self.label_to_id, label)
            counter += 1

        return sentence_vectors, one_hot_encoding_label

    def get_embedding_matrix(self):
        """
        load pre-trained embedding file and create embedding matrix
        """
        if self.config['PARAMETER']['embedding_selection'] == 'Random initialize':
            return None
        else:
            word2vec = load_embedding_dic(self.config['PATH']['embedding_path'])
            vocab_size = len(self.char_to_id)
            embedding_matrix = np.zeros((vocab_size, int(self.config['PARAMETER']['embedding_dim'])))
            for word, i in self.char_to_id.items():
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all zeros.
                    embedding_matrix[i] = embedding_vector
            return torch.from_numpy(embedding_matrix).float()
