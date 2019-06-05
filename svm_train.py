##########################
###  Organise Imports  ###
##########################

import numpy as np
import os
import glob
import cv2
import mahotas
from matplotlib import pyplot
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn import metrics

############################
### Load data and labels ###
############################

# fixed-sizes for image
fixed_size = tuple((331, 331))

# path to training data
base_dir = os.getcwd()

# train_test_split size
test_size = 0.20

# seed for reproducing same results
seed = 1

data = input("Type the file name of the training data: ")
label = input("Type the file name of the training labels: ")

# load data
features = np.load(os.path.join(base_dir, data))
labels = np.load(os.path.join(base_dir, label))

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(features.shape))
print("[STATUS] labels shape: {}".format(labels.shape))

# split the training and testing data
(train_data, test_data, train_labels, test_labels) = train_test_split(np.array(features), 
                                                                      np.array(labels),
                                                                      test_size=test_size,
                                                                      random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(train_data.shape))
print("Test data   : {}".format(test_data.shape))
print("Train labels: {}".format(train_labels.shape))
print("Test labels : {}".format(test_labels.shape))


###############################
### filter all the warnings ###
###############################

import warnings
warnings.filterwarnings('ignore')

####################
### Create model ###
####################

# create the SVM model
model = SVC(random_state=seed,kernel = 'linear')
model.fit(train_data,train_labels)
pred_labels = model.predict(test_data)
print("Classification report for model %s:\n%s\n" % (model, metrics.classification_report(test_labels, pred_labels)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, pred_labels))

# save the model to disk
filename = input("Save model as: ")
pickle.dump(model, open(filename, 'wb'))
