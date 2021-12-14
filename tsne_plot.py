"""
CSC2515 project 2021
Author: Barza Nisar

This file loads features from SSL models, plots t-SNE on these features and computes distances between t-SNE clusters.
"""
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pickle

from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def load_class_names():
    meta = pickle.load(open("../data/CIFAR10/cifar-10-batches-py/batches.meta", "rb"))
    return meta["label_names"]

def load_features(result_dir):
    X_trainval_feature = pickle.load(open(os.path.join(result_dir, "X_trainval_feature.pkl"), "rb"))
    y_trainval = pickle.load(open(os.path.join(result_dir, "y_trainval.pkl"), "rb"))
    return X_trainval_feature, y_trainval

# Compute distances between class clusters
def class_feature_distances(result_dir, X_trainval_feature, y_trainval, model):

    class_names = load_class_names()
    mean_rep_for_all_classes = np.zeros((len(class_names), X_trainval_feature.shape[1]))

    for idx, label in enumerate(class_names):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(y_trainval.tolist()) if l == idx]
        this_class_features = np.take(X_trainval_feature, indices, axis=0)
        mean_rep_for_all_classes[idx, :] = np.mean(this_class_features, axis=0)


    #for metric in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]:
    metric = "cosine"
    D = pairwise_distances(mean_rep_for_all_classes, metric=metric)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)  # for label size
    sns.heatmap(D, annot=True, annot_kws={"size": 12}, xticklabels=class_names,
                yticklabels=class_names)  # font size
    plt.title(f"{metric} distances between {model}'s tsne class centers on CIFAR10 ")
    plt.savefig(os.path.join(result_dir, f"{metric}.png"))
    #plt.show()

    #print mean, std and min distances between class cluster centers
    mean = np.mean(D[D > 0])
    std = np.std(D[D > 0])

    # Find most similar classes
    C = D
    C[D == 0] = 1000000
    argmin = np.unravel_index(C.argmin(), C.shape)
    min = C[argmin[0], argmin[1]]
    min_class_names = [class_names[argmin[0]], class_names[argmin[1]]]

    # Find second most similar classes
    C[argmin[0], argmin[1]] = 100000
    C[argmin[1], argmin[0]] = 100000
    argmin2 = np.unravel_index(C.argmin(), C.shape)
    min2 = C[argmin2[0], argmin2[1]]
    min2_class_names = [class_names[argmin2[0]], class_names[argmin2[1]]]

    # Find third most similar classes
    C[argmin2[0], argmin2[1]] = 100000
    C[argmin2[1], argmin2[0]] = 100000
    argmin3 = np.unravel_index(C.argmin(), C.shape)
    min3 = C[argmin3[0], argmin3[1]]
    min3_class_names = [class_names[argmin3[0]], class_names[argmin3[1]]]



    tsne_summary = {"mean_cluster_dist": mean,
                    "std_cluster_dist": std,
                    "min_cluster_dist": [min, min2, min3],
                    "most_similar_classes": [min_class_names, min2_class_names, min3_class_names]}

    print(tsne_summary)
    pickle.dump(tsne_summary, open(os.path.join(result_dir, "cosine_summary.pkl"), "wb"))


def visualize_tsne(result_dir, y_trainval, model, tsne):

    class_names = load_class_names()
    num_classes = len(class_names)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes))

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for idx, label in enumerate(class_names):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(y_trainval.tolist()) if l == idx]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label, marker='x', linewidth=0.5)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"TSNE on CIFAR10 for {model} features")

    filename = f"tsne_{model}.pdf"
    # finally, show the plot
    plt.savefig(os.path.join(result_dir, filename))
    #plt.show()

# Visualize t-SNE for all models on cifar10
models = ["byol", "moco-v2", "simclr-v2", "swav", "barlow", "supervised"]
for model in models:
    results_dir = os.path.join(os.getcwd(), "results", model, "cifar10", "None")
    X_trainval_feature, y_trainval = load_features(results_dir)

    #Compute tsne
    tsne = TSNE(n_components=2).fit_transform(X_trainval_feature)
    pickle.dump(tsne, open(os.path.join(results_dir, "tsne.pkl"), "wb"))

    #load tsne
    tsne = pickle.load(open(os.path.join(results_dir, "tsne.pkl"), "rb"))
    visualize_tsne(results_dir, y_trainval, model, tsne)
    class_feature_distances(results_dir, tsne, y_trainval, model)