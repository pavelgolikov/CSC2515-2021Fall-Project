#!/usr/bin/env python
# coding: utf-8
import os
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_data(model,dataset,augmentation):
    results_dir = os.path.join(os.getcwd(), "results", model, dataset, augmentation)
    y_test = pickle.load(open(os.path.join(results_dir, "y_test.pkl"), "rb"))
    y_pred_labels = pickle.load(open(os.path.join(results_dir, "y_pred_labels.pkl"), "rb"))
    return y_test, y_pred_labels

def compute_class_accuracy(y_test, y_pred_labels):
    cm = confusion_matrix(y_test, y_pred_labels)
    per_class_accuracies = np.nan_to_num(cm.diagonal() / cm.sum(axis=1))*100
    test_mean_class_acc = per_class_accuracies.mean()
    return per_class_accuracies, test_mean_class_acc

def plot_class_vs_class_accuracy(models_list, per_class_accuracies, dataset):
    plt.plot(per_class_accuracies,'.')
    plt.title('Comparison of models on class accuracies of ' +str(dataset)+ ' dataset')
    plt.xlabel("classes")
    plt.ylabel("Per_class_accuracy %")
    plt.legend(models_list)

def plot_model_vs_mean_class_accuracy(models_list, dataset_list, test_mean_class_acc):
    for i in dataset_list:
        plt.plot(models_list, test_mean_class_acc[:,dataset_list.index(i)], '.')
        plt.ylim([20, 100])
        plt.title("Mean class accuracies of models on different datasets")
        plt.xlabel("Model")
        plt.ylabel("Mean of per_class_accuracies %")
        plt.legend(dataset_list)
    plt.savefig('results\model_vs_mean_class_accuracy.jpg', dpi = 500, bbox_inches = 'tight')
    plt.show()

def main():
    models_list = ["supervised","moco-v2","simclr-v2","byol","swav","barlow"]
    dataset_list = ["aircraft","cars","cifar10"]
    augmentation = "None"
    
    test_mean_class_acc = np.zeros((len(models_list),len(dataset_list)))
    for dataset in dataset_list:
        for model in models_list:
            y_test, y_pred_labels = load_data(model,dataset,augmentation)
            per_class_accuracies, test_mean_class_acc[models_list.index(model),dataset_list.index(dataset)] = compute_class_accuracy(y_test, y_pred_labels)
            plot_class_vs_class_accuracy(models_list, per_class_accuracies, dataset)
        plt.savefig('results\ '+str(dataset)+'_class_accuracies.jpg', dpi = 500, bbox_inches = 'tight')
        plt.show()
    plot_model_vs_mean_class_accuracy(models_list, dataset_list, test_mean_class_acc)

if __name__ == '__main__':
    main()