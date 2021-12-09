# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:40:36 2021

@author: Pranav
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

#For training the linear classifier on these models, the batch size = 32 and the 100% datasets were used

#Function to obtain Overall model accuracy
def get_model_accuracy(model, dataset):
    path = r"results\{}\{}\None".format(model,dataset)
    Y_pred= open(os.path.join(os.getcwd(), path,'y_pred_labels.pkl'), 'rb')
    y_pred = pickle.load(Y_pred)
    Y_true = open(os.path.join(os.getcwd(), path,'y_test.pkl'), 'rb')
    y_true = pickle.load(Y_true)
    overall_model_acc = accuracy_score(y_true, y_pred)*100
    
    return overall_model_acc

#Function to plot the overall accuracies per model per dataset
def plot_overall_accuracy(Acc_df, directory):
    ax = Acc_df.plot.bar(rot=0)
    fig1 = ax.get_figure()
    fig1.set_size_inches(7, 6)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best", fontsize=9)
    ax.set_title('Overall Accuracy per Dataset for each SSL Model')
    ax.tick_params(axis='x', rotation=0)
    ax.set_xticklabels(Acc_df.Dataset)
    ax.set_yticks([i for i in range (100) if i%5 == 0])
    plt.savefig('{}/Overall_Acc_per_Dataset.jpg'.format(directory), dpi = 500, bbox_inches = 'tight')
    plt.show()
    
    

'''
df_pivot1 = pd.pivot_table(
	Overall_Acc_df,
	values="Overall Accuracy",
	index="Dataset",
	columns="Model")
ax = df_pivot1.plot(kind="bar")
fig1 = ax.get_figure()
fig1.set_size_inches(7, 6)
ax.set_xlabel("Dataset")
ax.set_ylabel("Accuracy")
ax.legend(loc="best", fontsize=9)
ax.set_title('Overall Accuracy per Dataset for each SSL Model')
ax.tick_params(axis='x', rotation=0)
ax.set_yticks([i for i in range (100) if i%5 == 0])
plt.show()
'''

#Function to obtain per class accuracy for each model
def get_perclass_acc(model, dataset):
    path = r"results\{}\{}\None".format(model,dataset)
    Y_pred= open(os.path.join(os.getcwd(), path,'y_pred_labels.pkl'), 'rb')
    y_pred = pickle.load(Y_pred)
    Y_true = open(os.path.join(os.getcwd(), path,'y_test.pkl'), 'rb')
    y_true = pickle.load(Y_true)
    cm = confusion_matrix(y_true, y_pred)
    acc_per_class = cm.diagonal()/cm.sum(axis=1)*100
    
    return acc_per_class


def main():
    models = ['moco-v2','simclr-v2','byol','swav','barlow','supervised']
    datasets = ['cifar10', 'cars', 'aircraft']

    Model_names = ['MoCo-v2','SimCLR-v2','BYOL', 'SWAV','Barlow Twins','Supervised']
    Data = ['CIFAR10', 'Cars', 'Aircraft']
    
    ##########################################
    #Overall Accuracy per model per dataset

    #cols = ["Dataset","Model","Overall Accuracy"]
    #Overall_Acc_df = pd.DataFrame(columns = cols)
    #for dataset in datasets:
        #for model in models:
            #model_acc= get_model_accuracy(model, dataset)
            #print('The final accuracy on the {} test set for {} is {} %'.format(dataset, model, model_acc))
            #row = {'Dataset': Data[datasets.index(dataset)],'Model': Model_names[models.index(model)],'Overall Accuracy':model_acc}
            #Overall_Acc_df=Overall_Acc_df.append(row, ignore_index=True)
    #print('\n')
    #print(Overall_Acc_df)
    
    #Tabulating the accuracies for visualization
    model_cols = ['Dataset','MoCo-v2','SimCLR-v2','BYOL','SWAV','Barlow Twins','Supervised']
    Overall_Acc_df = pd.DataFrame(columns = model_cols)
    for dataset in datasets:
        model_acc_list= []
        model_acc_list.append(Data[datasets.index(dataset)])
        for model in models:
            model_acc= get_model_accuracy(model, dataset)
            model_acc_list.append(model_acc)
            print('The final accuracy on the {} test set for {} is {} %'.format(Data[datasets.index(dataset)], 
                                                                                Model_names[models.index(model)], model_acc))
        df_series = pd.Series(model_acc_list, index = Overall_Acc_df.columns)
        Overall_Acc_df = Overall_Acc_df.append(df_series, ignore_index=True)
        print('\n')
    
    print(Overall_Acc_df)
    
    ootd_dir = os.path.join(os.getcwd(), "out_of_distribution_testing")
    if not os.path.exists(ootd_dir):
        os.makedirs(ootd_dir)
        
    pickle.dump(Overall_Acc_df, open(os.path.join(ootd_dir, "Overall_Acc_df.pkl"), "wb"))
    
    plot_overall_accuracy(Overall_Acc_df, ootd_dir) #Plots the graph and saves the plot in out_of_distribution_testing_library
    print('\n')
    
    #####################################################################
    #Per class accuracy for each model and dataset
    
    #Obtain and plot per class accuracies for each model
    cols2 = ["Dataset","Model","Class Accuracies"]
    PerClass_Acc_df = pd.DataFrame(columns = cols2)
    for dataset in datasets:
        for model in models:
            model_acc_pc= get_perclass_acc(model, dataset)
            row_1 = {'Dataset': Data[datasets.index(dataset)],'Model': Model_names[models.index(model)],'Class Accuracies': model_acc_pc}
            PerClass_Acc_df = PerClass_Acc_df.append(row_1, ignore_index=True)
            X = np.arange(len(model_acc_pc))
            Y = model_acc_pc
            Num_classes = len(model_acc_pc)
            plt.plot(X, Y, label = Model_names[models.index(model)], marker=".")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("Class (Total Number of Classes: {})".format(Num_classes))
            plt.ylabel("Accuracy")
            plt.title('Accuracy per Class for "{}" dataset'.format(Data[datasets.index(dataset)]))
            plt.savefig('{}/Per_Class_Acc_{}.png'.format(ootd_dir,dataset), dpi = 500, bbox_inches = 'tight')
        
        #plt.tight_layout()
        plt.show()
        plt.figure(figsize=(10, 3))
    
    print(PerClass_Acc_df)
    pickle.dump(PerClass_Acc_df, open(os.path.join(ootd_dir, "PerClass_Acc_df.pkl"), "wb"))
    

if __name__ == '__main__':
    main()