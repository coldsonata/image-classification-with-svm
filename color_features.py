##########################
###  Organise Imports  ###
##########################

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import h5py


# filter all the warnings
import warnings
warnings.filterwarnings('ignore')



##########################
### Generate constants ###
##########################

# fixed-sizes for image
fixed_size = tuple((331, 331))

# path to training data
base_dir = os.getcwd()

# train_test_split size
test_size = 0.20

# seed for reproducing same results
seed = 1

# get the training labels
train_label1 = input("Folder name of first group (Case Sensitive): ")
train_label2 = input("Folder name of second group (Case Sensitive): ")
train_labels = [train_label1, train_label2]

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# num of images per class
images_per_class = 4000

# choose save name
save_name = input("Choose the name of the file that the features will be saved as: ")



##########################################
### Create feature extractor functions ###
##########################################

# color-feature-descriptor: BGR Histogram
def bgr_histogram(image, mask=None):
    # compute the color histogram
    b_hist  = (cv2.calcHist([image], [0], None, [64], [0, 256])).flatten()
    g_hist = (cv2.calcHist([image], [1], None, [64], [0, 256])).flatten()
    r_hist = (cv2.calcHist([image], [2], None, [64], [0, 256])).flatten()
    # normalize the histogram
    cv2.normalize(b_hist, b_hist)
    cv2.normalize(g_hist, g_hist)
    cv2.normalize(r_hist, r_hist)
    # combine the histograms
    hist = np.hstack([b_hist, g_hist, r_hist])
    # return the histogram
    return hist

# color-feature-descriptor: HSV Histogram
def hsv_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [16, 4, 4], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# color-feature-descriptor: LAB Histogram
def lab_histogram(image, mask=None):
    # convert the image to LAB color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [4, 14, 14], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
	
	

	
	
	
	
	
###########################
### Extracting features ###	
###########################	

print("[STATUS] Extracting features... This may take a while")
	
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(base_dir, training_name)

    # get the current training label
    current_label = training_name
    
    num_images = 0
    index = 0  
    # loop over the images in each sub-folder
    while (num_images < images_per_class):
        # get the image file name
        file = os.listdir(dir)[index]
        index += 1
        
        # read the image and resize it to a fixed-size
        image = cv2.imread(os.path.join(dir,file))
        try:
            image = cv2.resize(image, fixed_size)
            num_images += 1
        except: 
            continue

        ####################################
        # Color Feature extraction
        ####################################
        bgr = bgr_histogram(image)
        hsv   = hsv_histogram(image)
        lab  = lab_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([bgr, hsv, lab])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")




#####################
### Save features ###
#####################

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector 
np.save(os.path.join(base_dir, save_name + '_data'),np.array(rescaled_features))
np.save(os.path.join(base_dir, save_name + '_labels'),np.array(target))

print("[STATUS] end of preprocessing..")