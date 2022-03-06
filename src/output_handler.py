import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def transform_output(one_hot_predict, one_hot_true):
    """
    transform one-hot encoding tensor to numpy int label
    """
    prediction = one_hot_predict.cpu().detach().numpy()  # convert tensor to numpy
    prediction = np.argmax(prediction, axis=1)  # select the largest probability as the prediction label
    true = one_hot_true.cpu().detach().numpy()
    true = np.argmax(true, axis=1)
    return prediction, true


def get_id_to_label(path):
    """
    create a id_to_label dictionary mapping id to label
    """
    with open(path, "rb") as f:
        label_to_id = pickle.load(f)
    id_to_label = {value: key for key, value in label_to_id.items()}
    return id_to_label


def get_classes(int_predict, int_true, id_to_label):
    """
    get the text classes used for plotting confusion matrix and displaying f1 score of each class
    """
    int_classes = list(set(list(int_true) + list(int_predict)))
    text_classes = []
    for int_class_item in int_classes:
        text_class_item = id_to_label.get(int_class_item + 1)
        text_classes.append(text_class_item)
    return text_classes


def evaluate_during_training(one_hot_predict, one_hot_true):
    """
    used for evaluate the model during training
    """
    # transform the output from model
    int_predict, int_true = transform_output(one_hot_predict, one_hot_true)
    # calculate the accuracy and f1 score
    accuracy = accuracy_score(int_predict, int_true)
    weighted_f1 = f1_score(int_true, int_predict, average='weighted')
    return weighted_f1


def plot_confusion_matrix_figure(true, predict, classes, save_img_path):
    """
    plot, display and save the confusion matrix
    """
    confusion_matrix_result = confusion_matrix(true, predict)
    plt.figure(figsize=(30, 30))
    plt.imshow(confusion_matrix_result, interpolation='nearest', cmap=plt.cm.Oranges)
    indices = range(len(confusion_matrix_result))
    plt.xticks(indices, classes, rotation=90)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Predict')
    plt.ylabel('True')
    for first_index in range(len(confusion_matrix_result)):
        for second_index in range(len(confusion_matrix_result[first_index])):
            plt.text(first_index, second_index, confusion_matrix_result[first_index][second_index],
                     ha='center', va='center')
    plt.savefig(fname=save_img_path)
    plt.show()


class OutputHandler:
    def __init__(self, config, raw_sentence):
        self.config = config
        self.accuracy = 0
        self.micro_f1 = 0
        self.macro_f1 = 0
        self.weighted_f1 = 0
        self.f1_each_class = None
        self.confusion_matrix_result = None
        self.raw_sentence = raw_sentence
        self.int_true = None
        self.int_predict = None
        self.text_classes = None
        self.id_to_label = get_id_to_label(self.config['PATH']['label_to_id_path'])

    def result_evaluation(self, one_hot_predict, one_hot_true):
        """
        Evaluate the model:
        (1) calculate the accuracy, three types of F1 score (micro, macro, weighted)
        (2) F1 score for each class
        (3) calculate the confusion matrix
        """
        int_predict, int_true = transform_output(one_hot_predict, one_hot_true)
        self.int_true = int_true
        self.int_predict = int_predict
        # calculate the accuracy and f1 score
        self.accuracy = accuracy_score(int_predict, int_true)
        self.micro_f1 = f1_score(int_true, int_predict, average='micro')
        self.macro_f1 = f1_score(int_true, int_predict, average='macro')
        self.weighted_f1 = f1_score(int_true, int_predict, average='weighted')
        self.f1_each_class = f1_score(int_true, int_predict, average=None)
        self.text_classes = get_classes(int_predict, int_true, self.id_to_label)
        plot_confusion_matrix_figure(int_predict, int_true, self.text_classes,
                                     self.config['PATH']['confusion_martix_path'])
        self.confusion_matrix_result = confusion_matrix(int_true, int_predict)
        print("Accuracy on the test set:", self.accuracy)
        print("Micro F1 score on the test set:", self.micro_f1)
        print("Macro F1 score on the test set:", self.macro_f1)
        print("Weighted F1 score on the test set:", self.weighted_f1)

    def write_result(self):
        """
        write the evaluation results
        (1) accuracy and total three types of f1 score (micro, macro, weighted)
        (2) f1 score of each class
        (3) write the original sentence, prediction label and true label for each testing question
        """
        class_f1_dic = {}
        for class_item, f1_item in zip(self.text_classes, self.f1_each_class):
            class_f1_dic[class_item] = f1_item
        class_f1_sorted = sorted(class_f1_dic.items(), key=lambda kv: (kv[1], kv[0]))

        max_length = max([len(s) for s in self.raw_sentence])
        with open(self.config['PATH']['result_path'], 'w') as result:
            result.write('Accuracy on test set: ' + str(self.accuracy))
            result.write('\n')
            result.write('Micro F1-score on test set: ' + str(self.micro_f1))
            result.write('\n')
            result.write('Macro F1-score on test set: ' + str(self.macro_f1))
            result.write('\n')
            result.write('Weighted F1-score on test set: ' + str(self.weighted_f1))
            result.write('\n\n')
            for item in class_f1_sorted:
                result.write(item[0] + ': ' + str(item[1]))
                result.write('\n')
            result.write('\n')
            space = " " * (max_length - 1 - len('Sentence'))
            result.write('Sentence' + space + '(True label, Prediction)')
            result.write('\n')
            for s, t, p in zip(self.raw_sentence, self.int_true, self.int_predict):
                space = " " * (max_length + 1 - len(s))
                line = s[1:-1] + space + '(' + self.id_to_label.get(t + 1) + ', ' + self.id_to_label.get(p + 1) + ')'
                result.write(line)
                result.write('\n')
            result.close()
