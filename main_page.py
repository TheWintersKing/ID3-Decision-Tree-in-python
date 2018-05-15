import numpy as np
import pickle
import csv
import Tree_builder as ct
import Cross_valid as cv
import test_splitter as ts 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from pprint import pprint

def main():
    header = input ("enter 'attribute name' file name and location with extension (only .csv): \n")
    name = input ("enter location and file name of 'data' with  extension (only.csv):  \n")
    with open(header) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for columns in csvReader:
            columns = columns
    dataset = pd.read_csv (name,header = None,names = columns)
    train , test = ts.test_split (dataset)
    score2 = cv.evaluate_algorithm (train , columns)
    score = ct.start_decision_tree (train , test , True)
    print ("Valiation Score : %f \n Test Score : %f" %(score2 , score))
    return






main()
