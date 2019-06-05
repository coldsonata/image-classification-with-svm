##########################
###  Organise Imports  ###
##########################

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import mahotas
import math
from skimage.feature import hog as sk_hog





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

# line-feature-descriptor
def line_features(image):
    # detect edges of image with Canny Detector
    edges = cv2.Canny(image,100,200)
    # apply Hough Line Transformation to extract lines detected
    lines = cv2.HoughLines(edges,1,np.pi/180,80)
    return lines

def get_straight_lines_number(lines):
    if lines is not None:
        return len(lines)
    else:
        return 0

def get_percent_parallel(lines):
    if lines is not None:
        empty_list = []
        parallel_lines = 0
        for line in lines:
            # get the angle
            angle = line[0][1]
            # check if r is negative
            if line[0][0] < 0:
                angle = round(np.pi - angle,7)
            # check if angle is the first of its kind
            if angle not in empty_list:
                empty_list.append(angle)
            else:
                parallel_lines += 1
        return parallel_lines / get_straight_lines_number(lines)
    else:
        return 0

def get_histogram_of_orientations(lines):
    if lines is None:
        return np.zeros(9)
    list = []
    for line in lines:
        # get the angle
        angle = line[0][1]
        # check if r is negative
        if line[0][0] < 0:
            angle = round(np.pi + angle,7)
        # add to the list of angles
        list.append(angle)
    # create the histogram
    hist_orientation, x = np.histogram(list,bins = 9,range = (0, np.pi))
    #normalize the output
    hist_orientation = hist_orientation.astype(float)
    cv2.normalize(hist_orientation,hist_orientation)
    return hist_orientation

def get_histogram_of_distances(lines):
    if lines is None:
        return np.zeros(6)
    list = []
    for line in lines:
        # get the magnitude
        mag = line[0][0]
        # check if r is negative
        if line[0][0] < 0:
            mag = -1*mag
        # add to the list of magnitudes
        list.append(mag)
    # create the histogram
    hist_orientation, x = np.histogram(list,bins = 6, range = (0,468))
    #normalize the output
    hist_orientation = hist_orientation.astype(float)
    cv2.normalize(hist_orientation,hist_orientation)
    return hist_orientation

def get_global_corners(image):
    # detect corners of image
    corners = cv2.cornerHarris(image,2,5,0.001)
    # get threshold
    max = 0.1*corners.max()
    # count corners
    count = 0
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if corners[i,j] >= max:
                count += 1
    return count/115600
    
def get_local_corners(image):
    # pad image to 340 x 340
    top, bottom = 4, 5
    left, right = 4, 5
    color = [0, 0, 0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # split image into 20x20, we now have a list of 20x20 images
    img_lst = []
    for i in range(17):
        x = image[20*i:20*(i+1)]
        for j in range(17):
            new_im = []
            for array in x:
                new_im.append(array[20*j:20*(j+1)])
            new_im = np.array(new_im)    
            img_lst.append(new_im)
    
    # for each image fragment, detect the corners of it
    corner_ratio = 0
    for img in img_lst:
        corner_ratio += get_global_corners(img)
    return corner_ratio
        

def get_edge_orientation_hist(image):
    # create edges for horizontal and vertical
    h = cv2.Sobel(image,-1,1,0,3)
    v = cv2.Sobel(image,-1,0,1,3)
    h = h.flatten()
    v = v.flatten()
    array = np.append(h,v)
    hist, x = np.histogram(array,bins = 64, range = (0,255))
    # normalize the output
    hist = hist.astype(float)
    cv2.normalize(hist,hist)
    return hist

def get_hog(image):
    # split image into 66x66
    img_lst = []
    for i in range(5):
        if (i == 5):
            x = image[20*i:]
        else :
            x = image[20*i:20*(i+1)]
        for j in range(5):
            new_im = []
            if (j == 5):
                for array in x:
                    new_im.append(array[20*j:20*(j+1) + 1])
            else :
                for array in x:
                    new_im.append(array[20*j:20*(j+1)])
            new_im = np.array(new_im)    
            img_lst.append(new_im)
            
    features = []
    # for each image
    for im in img_lst:
        # Calculate gradient 
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        max_angle = 180
        nbins = 9
        # calculate histogram of gradients, with reference to and adapting code from
        # https://github.com/JeanKossaifi/python-hog/blob/master/hog/histogram.py
        b_step = max_angle/nbins
        b0 = (angle % max_angle) // b_step
        b1 = b0 + 1
        b1[np.where(b1>=nbins)]=0
        b = np.abs(angle % b_step) / b_step
        # create the arrays of zeros
        hist = np.zeros(nbins)
        for i in range(nbins):
            hist[i] += sum(np.where(b0 == i,(1-b)*magnitude,0).flatten())
            hist[i] += sum(np.where(b1 == i,(1-b)*magnitude,0).flatten())
        hist = cv2.normalize(hist,hist)
        features.append(hist)
        
    return (np.array(features)).flatten()
    

	
	
	
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
        # Shape Feature extraction
        ####################################
        
        # extract lines
        lines = line_features(image)
        
        # generate features
        s_lines = get_straight_lines_number(lines)
        pp_lines = get_percent_parallel(lines)
        hist_o = get_histogram_of_orientations(lines)
        hist_d = get_histogram_of_distances(lines)
        g_corners = get_global_corners(image)
        l_corners = get_local_corners(image)
        eod = get_edge_orientation_hist(image)
        hog = get_hog(image)
        
        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([[s_lines], [pp_lines], hist_o, hist_d, [g_corners], [l_corners], eod, hog])

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