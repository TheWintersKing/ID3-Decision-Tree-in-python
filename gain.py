import numpy as np
import pandas as pd 

def gain( s , x, y):
	threshold = None
	if s[x].dtype == 'float16' or s[x].dtype == 'float32' or s[x].dtype == 'float64':
		threshold , res  = find_threshold_and_gain(s,x,y)
	else :
		res = discret_information_gain (s , x , y)
	return res,threshold

def discret_information_gain (s, x , y):
	res = entropy(y)
	val, counts = np.unique(s.loc[:,x], return_counts=True)
	freqs = counts.astype('float')/len(y)
	for p, v in zip(freqs, val):
		res -= p * entropy(s[(s[x]==v)].loc[:,'class'])
	return res

def continous_information_gain (s,x,y,t):
	res = entropy(y)
	v = s.loc[:,x]
	counts  = np.array([v[v < t].count() , v[v>=t].count()])
	#print (counts)
	freqs = counts.astype('float')/len(y)
	res -= freqs[0] * entropy(s[(s[x]<t)].loc[:,'class']) - freqs[1] * entropy(s[(s[x]>=t)].loc[:,'class'])
	return res

def entropy (s):
	res = 0
	val, counts = np.unique(s, return_counts=True)
	freqs = counts.astype('float')/len(s)
	for p in freqs:
		if p != 0.0:
			res -= p * np.log2(p)
	return res

def find_threshold_and_gain (s , x, y):
	copy = s
	val, counts = np.unique(s.loc[:,x], return_counts=True)
	val.sort()
	threshold = []
	for i in range (0 , len(val)-1):
		threshold.append ((val[i] + val[i+1])/2)
	#print (threshold)
	threshold = set(threshold)
	#print (threshold)
	for v in threshold :
		if any(s.loc[:,x]==v)  :
			threshold.remove(v)
	thresholdtry = []
	IG = []
	for t in threshold:
		IG.append(continous_information_gain(s,x,y,t))
		thresholdtry.append(t) #f]
	#print (IG)
	if not IG:
		return None , None
	maxIG=max(IG)
	maxThresh=IG.index(maxIG)

	return thresholdtry[maxThresh] , maxIG