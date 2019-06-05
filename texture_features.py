##########################
###  Organise Imports  ###
##########################

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel




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

# create gabor kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


##########################################
### Create feature extractor functions ###
##########################################

def get_lbp(image):
    lbp = local_binary_pattern(image, 8, 1, method="nri_uniform")
    hist, bins = np.histogram(lbp,bins = 59,range = (0,59))
    return hist

def get_gabor_features(image):
    features = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        features[k, 0] = filtered.mean()
        features[k, 1] = filtered.var()
    return (cv2.normalize(features,features)).flatten()

	
	
	
	
	
	
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
        image = cv2.imread(os.path.join(dir,file),0)
        try:
            image = cv2.resize(image, fixed_size)
            num_images += 1
        except: 
            continue
        
        ####################################
        # Texture Feature extraction
        ####################################
        
        # generate features
        lbp = get_lbp(image)
        gabor = get_gabor_features(image)
        
        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([lbp, gabor])

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