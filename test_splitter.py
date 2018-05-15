import pandas as pd 
import numpy as np 

def test_split (t) :
    train = t.sample(frac = 0.8 , replace = False)
    return train , t