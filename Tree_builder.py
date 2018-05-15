import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from collections import OrderedDict
from pprint import pprint
import gain as gx
import re


def predict (tree , row , res = 'class'):
    for key , val in tree.items ():
        attr ,job, cut = get_attr (key)
        if job == '=':
            if str(row[attr]) == cut :
                if isinstance (val , dict):
                    return  predict (val , row)
                elif val ==  row['class']:
                    return  1
                else:
                    return 0
        elif job == '<':
            if float(row[attr]) < float(cut) :
                if isinstance (val , dict):
                    return  predict (val , row)
                elif val == row['class']:
                    return  1
                else:
                    return 0
        if job == '>=':
            if float(row[attr]) == float(cut) :
                if isinstance (val , dict):
                    return  predict (val , row)
                elif val ==  row['class']:
                    return  1
                else:
                    return 0
    return 0


def get_attr (key):
    attr = key.split(" ")
    for a in attr:
        if a == '':
            attr.remove(a)
    #print(attr)
    return attr[0] ,attr[1], attr[2] 

def start_decision_tree (train , test , printer) :
    tree = build_tree (train)
    pn = list()
    logFile=open('mylogfile'+'.txt', 'w')
    if printer :
        pprint (tree , width = 1)
        pprint(tree , width = 1 , stream= logFile)
    logFile.close()
    for index, row in test.iterrows():     
        pn.append(predict (tree , row))
    #print_list_tree(tree)
    #print (train.dtypes)
    #print (tree_to_rules(tree))
    return sum (pn) / float(len (test)) * 100

def build_tree (dataset):
    
    tree = {}

    cont = None
    spin = 0

    y = class_find (dataset)

    if is_pure(dataset):
            return most_prob (y)

    col = [(format(x_attr), gx.gain(dataset, format(x_attr), y)) for x_attr in dataset.columns if format(x_attr) != 'class']

    gainx = 1e-6

    for c in col:
        if c[1][0] > gainx :
            gainx = c[1][0]
            if c[1][1] is None:
                split_attr = c[0]
            else :
                split_attr = c[0]
                cont = c[1][1]
    
    if gainx == 1e-6 :
        return most_prob (y)
        
    #print (dataset.loc[:,split_attr].head(5))
    
    for subt in create_subsets(dataset, split_attr , cont):
        if cont is None:
            v = subt[split_attr].iloc[0]
            tree["%s = %s" % (split_attr, v)] = build_tree(subt.drop(columns = [split_attr]))
        else :
            v = cont
            if spin == 0 :
                tree["%s < %s" % (split_attr, v)] = build_tree(subt.drop(columns = [split_attr]))
                spin = 1
            else :
                tree["%s >= %s" % (split_attr, v)] = build_tree(subt.drop(columns = [split_attr]))
                spin = 0
    return tree

def most_prob (s):
    val, counts = np.unique(s, return_counts=True)
    i = np.argmax(counts)
    return val[i]

def class_find(s) :
    return s.loc[: , 'class']

def is_pure (s):
    return len(unique_nodes (s , 'class')) == 1 or len(class_find(s)) == 0

def create_subsets (s , x , cont):
    if cont is None:
        return [s[(s[x] == v)].loc[:] for v in unique_nodes(s , x)]
    else :
        return [s[(s[x] < cont)].loc[:] , s[(s[x] >= cont)].loc[:]]
        

def unique_nodes (s , x):
    return np.unique(s.loc[:,x], return_counts=False)


