import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def transform_output(prediction, true):
    """
    # transform one-hot encoding tensor to numpy int label
    """
    prediction = prediction.detach().numpy()  # convert tensor to numpy
    prediction = np.argmax(prediction, axis=1)  # select the largest probability as the prediction label
    true = true.detach().numpy()
    true = np.argmax(true, axis=1)
    return prediction, true


def calculate_evaluation_results(prediction, true):
    # transform one-hot encoding tensor to numpy int label
    predict, true = transform_output(prediction, true)
    # calculate the accuracy and f1 score
    accuracy = accuracy_score(true, predict)
    f1 = f1_score(true, predict, average='weighted')

    return accuracy, f1


def get_id_to_label(path):
    """
    create a id_to_label dictionary mapping id to label
    """
    with open(path, "rb") as f:
        label_to_id = pickle.load(f)
    id_to_label = {value: key for key, value in label_to_id.items()}
    return id_to_label


def write_result(config, sentence, true, prediction, f1, accuracy):
    """
    write the evaluation results
    (1) original sentence, true label and prediction for each testing question
    (2) the overall performance on the test set (accuracy and f1)
    """
    id_to_label = get_id_to_label(config['PATH']['label_to_id_path'])
    max_length = max([len(s) for s in sentence])
    with open(config['PATH']['result_path'], 'w') as result:
        result.write('Accuracy on testing set: ' + str(accuracy))
        result.write('\n')
        result.write('F1-score on testing set: ' + str(f1))
        result.write('\n\n')
        space = " " * (max_length - 1 - len('Sentence'))
        result.write('Sentence' + space + '(True label, Prediction)')
        result.write('\n')
        for s, t, p in zip(sentence, true, prediction):
            space = " " * (max_length + 1 - len(s))
            line = s[1:-1] + space + '(' + id_to_label.get(t + 1) + ', ' + id_to_label.get(p + 1) + ')'
            result.write(line)
            result.write('\n')
        result.close()


def result_evaluation(config, sentence, true, predict):
    print('-------------------Start testing-------------------')
    # calculate accuracy and f1 score
    accuracy, f1 = calculate_evaluation_results(predict, true)
    print('Accuracy on testing set:', accuracy)
    print('F1 score on testing set:', f1)
    # write the evaluation results
    predict, true = transform_output(predict, true)
    write_result(config, sentence, true, predict, f1, accuracy)
