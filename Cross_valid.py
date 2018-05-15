import numpy as np
import pickle
import csv
import Tree_builder as ct
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from pprint import pprint

def evaluate_algorithm (dataset, columns , n_folds = 3):
    print (len(dataset))
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    index = 0
    for fold in folds :
        train_set = None
        for tfold in folds :
            if not (tfold.equals(fold)):
                if train_set is None:
                    train_set = tfold
                    #print ("hERE : " , train_set.head(5))
                else:
                    train_set = pd.concat([train_set , tfold] , axis=0)
        obs_class = fold.loc[: , 'class']
        test_set = fold
        #print (train_set.dtypes)
        scores.append(ct.start_decision_tree (train_set , test_set , False))
        #print (scores)
        #scores.append(accuracy_cal(obs_class , obs_class , index))
        index += len(obs_class)
        valid = sum(scores)/len(scores)
    return valid

def cross_validation_split(dataset , n_folds):
    dataset_split = list()
    dataset_copy = dataset
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        if len (dataset_copy) < fold_size :
            fold = dataset_copy
        else :
            fold = dataset_copy.sample(n = fold_size , replace = False)
            #print (dataset_copy.head(5))
        dataset_split.append(fold)
    return dataset_split

def accuracy_cal(actual, predicted,index):
    correct = 0
    tick = list ()
    for i in range(0,len(actual)) :
        if actual.iloc[i] == predicted.iloc[i] :
            correct += 1
            #tick.append(1)
#    print (len(tick))
#    result = pd.merge(actual, predicted, on = 'class' , how='inner')
#    print (result.head(5))
#    correct = len(result)
    return correct / float(len(actual)) * 100.0