import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

# LOAD DATA FROM CSV FILE
cell_dataframe = pd.read_csv('cell_samples.csv')
# includes 700 records/rows and 10 columns/attributes/dimensions | target = to be predicted

# DISTRIBUTION OF CLASSES
benign_dataframe = cell_dataframe[cell_dataframe['Class'] == 2][0:200] # select class attribute (dictionary key) and access first 200 rows under column
malignant_dataframe = cell_dataframe[cell_dataframe['Class'] == 4][0:200] 

axes = benign_dataframe.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='Benign')
malignant_dataframe.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='Malignant', ax=axes)

# IDENTIFY UNWANTED COLUMNS
cell_dataframe = cell_dataframe[pd.to_numeric(cell_dataframe['BareNuc'], errors='coerce').notnull()] # checks for rows that have this attribut
cell_dataframe['BareNuc'] = cell_dataframe['BareNuc'].astype('int') # converts to integer

# REMOVE UNWANTED COLUMNS
feature_dataframe = cell_dataframe[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_dataframe) # convert dataframe to numpy array
# NOTE: x denotes Independent variable -> contributes to making prediction of dependent variable
y = np.asarray(cell_dataframe['Class']) # dependent variable -> whether it is benign or malignant

# DIVIDE THE DATA AS TRAIN/TEST DATASET
'''
cell_df ➡ Train/Test
Train(X, y) ➡ X is a 2D array
Test(X, y) ➡ y is a 1D array

'''
# y dataset is the class column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4) # chooses which rows to include in training data and the rest will be test data
# returns 4 arrays (X component of training and testing set and y component of testing and training set)

# MODELING (SVM WITH SCIKIT-LEARN)
# calculate distance from hyperplane
classifier = svm.SVC(kernel='linear', gamma='auto', C=2) # visualizes data in different dimension to help fit a hyperplane when agent could not originally fit it
# 2 units of penalty (C) for wrongly placed data points

classifier.fit(X_train, y_train) # X is uppercase because it holds multidimensional data # y holds single dimensional data

y_predict = classifier.predict(X_test) # makes prediction based on testing component

'''
SVM algorithm allows for several kernel functions for performing its processing (mapping data into higher dimensional space = kernelling).
Mathematical function used for transformation = kernel function.

Types of Kernels:
- Linear
- Polynomial
- Radial Basis Function (RBF)
- Sigmoid

Just test each one to see which works best.
'''

'''
Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
|
|   Current default is 'auto' which uses (1 / n_features),
|   if gamma='scale' is passed then it uses 1/ (n_features * X.var())
|   as value of gamma.
'''

# EVALUATION (RESULTS)
# compare y_prediction to y_test
print(classification_report(y_test, y_predict)) # compare predicted value vs test values

'''
Precisions = True Positive / True Positive + False Positive = True Positive / Total Predicted Positive # evaluates number of correct predictions
Recall = True Positive / True Positive + False Negative = True Positive / Total Actual Positive
F1 = 2 × (Precision × Recall / Precision + Recall)

Support = how may instances of certain numerical class (ex. 2 vs 4 for benign and malignant)
'''

plt.show()

# NOTE: whenever unsure about how to use a method or keyword in PYTHON, you can use help() method to give details on how to use method