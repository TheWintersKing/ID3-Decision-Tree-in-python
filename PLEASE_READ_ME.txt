**************************** READ ME****************************************************************
*			Assignment                                                                             *
****************************************************************************************************
-Coded in python3

-Uses ID3 Tree with discretization for continous values 

-Code consist of mutilpe files

-Too run compile main_page.py in python3

-The input to your program should consist of two files:
	
	1. 
	A file containing the name of the attributes. This file will contain the name of all the attributes in a comma separated format and the class attribute corresponds to the class.
	
	2.
	A comma separated file with all the samples. Each sample in a row of the file.

-The output of program are:
	
	1.
	Validiation Score

	2.
	Test Score

	3.
	Tree in test format (note the tree will also saved into a test file)

-Code can handle continous and discrete value (haven't tested on mutlivariant data)

-Use percentage cal to evaluate accuracy

-external libraries used numpy and pandas

- Cross Validation is hard coded to 3 folds(can be changed)

- Tree , cross validiation , test split were implement from scratch

- Files perfom
	$ (Tree_builder.py builds ID3 and makes prediction)
	$ (test_splitter.py split into training and test set)
	$ (cross_valid.py perform k_fold cross validiation)
	$ (gain.py Calculates gain used to split tree + calculates a threshold value used in discretization)

###(data sets and screenshots used are also included)###

****************************************************************************************************
