import pickle
import os
import numpy as np


def get_norm_weights():
    """Returns normalized (across models) matrix of class accuracies on training set. Elements will be used as 
    weights for computing ensemble prediction.
    
    Returns:
        ndarray: normalized matrix of class accuracies.
    """
    w_m = np.empty([10, 1])
    for model_name in ["barlow", "byol", "moco-v2", "simclr-v2", "swav"]:
        model_csv_file = os.path.join(os.getcwd(), 'results/ensemble/' + model_name + '_class_train_acc.csv')
        with open(model_csv_file, 'r') as f: # open in readonly mode
            # do your stuff
            mtca = np.loadtxt(f, delimiter=",")
            mtca = np.expand_dims(mtca, axis=1)
            # print(mtca)
            w_m = np.concatenate((w_m, mtca), axis=1)
            f.close()
    w_m = w_m[:,1:]
    
    row_sums = w_m.sum(axis=1)
    norm_w_m = w_m / row_sums[:, np.newaxis]
    dict_by_model = {
        "barlow": norm_w_m[:, 0],
        "byol": norm_w_m[:, 1],
        "moco-v2": norm_w_m[:, 2],
        "simclr-v2": norm_w_m[:, 3],
        "swav": norm_w_m[:, 4]
        }
    return dict_by_model


def get_pred_prob_y_labels():
    """Returns class probability vectors and y_labels on test set.
    
    Returns:
        (dictionary, dictionary): tuple of dictionaries corresponding to probability vectors are y_labels respectively.
    """
    all_pred_prob = {}
    # iterate over all subdirectories
    for model_name in ["barlow", "byol", "moco-v2", "simclr-v2", "swav"]:
        # paths to predictec probabilities
        prob_filename = os.path.join(os.getcwd(), 'results/' + model_name + '/cifar10/None/y_pred_probabilities.pkl')
        pred_prob = pickle.load(open(prob_filename, "rb"))
        all_pred_prob[model_name] = pred_prob
    
    # get test labels
    labels_filename = os.path.join(os.getcwd(), 'results/' + model_name + '/cifar10/None/y_test.pkl')
    y_labels = pickle.load(open(labels_filename, "rb"))
    
    return all_pred_prob, y_labels
    

def ensemble_prediction(norm_weights, y_pred_prob, y_labels):
    """Returns the accuracy of the ensemble on the test set.
    
    Args:
        norm_weights (ndarray): matrix of normalized training accuracies per class - acting as weights.
        y_pred_prob (ndarray): numpy array of prediction probabilities on the test set.
        y_labels (ndarray): numpy array of test set labels.
    """
    correct = 0
    per_class_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    per_class_correct_per_model = {
        'barlow': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'byol': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'moco-v2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'simclr-v2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'swav': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    per_class_totals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # iterate over predicted probabilities
    for i in range(len(y_pred_prob["barlow"])):
        per_class_totals[y_labels[i]] += 1
        pred = np.zeros((1, 10))
        # for each model
        for key in y_pred_prob.keys():
            # get weighted prediction for this model and this sample
            if y_labels[i] == np.argmax(y_pred_prob[key][i]):
                per_class_correct_per_model[key][y_labels[i]] += 1
            w_pred = np.multiply(norm_weights[key], y_pred_prob[key][i])
            norm = np.linalg.norm(w_pred)
            w_pred = w_pred / norm
            w_pred = w_pred / w_pred.sum()
            pred = pred + w_pred
        
        if y_labels[i] == np.argmax(pred):
            correct += 1
            per_class_correct[y_labels[i]] += 1
    # print accuracy
    print("Mean per class accuracy:")
    print(correct / len(y_pred_prob['barlow']) * 100)
    print('Per class accuracy:')
    print([per_class_correct[i] / per_class_totals[i] for i in range(10)])
    print("Per class per model:")
    # print(per_class_correct_per_model)
    print([x / 1000 for x in per_class_correct_per_model['barlow']])
    print([x / 1000 for x in per_class_correct_per_model['byol']])
    print([x / 1000 for x in per_class_correct_per_model['moco-v2']])
    print([x / 1000 for x in per_class_correct_per_model['simclr-v2']])
    print([x / 1000 for x in per_class_correct_per_model['swav']])


if __name__ == "__main__":
    # get normalized weights by model
    norm_weights = get_norm_weights()
    
    # get tensors with probabilities
    all_pred_prob, all_y_labels = get_pred_prob_y_labels()
    # get ensemble predictions on the test set
    ensemble_prediction(norm_weights, all_pred_prob, all_y_labels)