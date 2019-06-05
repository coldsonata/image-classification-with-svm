##########################
###  Organise Imports  ###
##########################

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import mahotas
import math
from skimage.feature import hog as sk_hog
from skimage.feature import local_binary_pattern
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage import color
import h5py






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
        image = cv2.imread(os.path.join(dir,file))
        try:
            image = cv2.resize(image, fixed_size)
            num_images += 1
        except: 
            continue
        
        ####################################
        # Texture Feature extraction
        ####################################
        
        # generate features
        bgr = bgr_histogram(image)
        hsv   = hsv_histogram(image)
        lab  = lab_histogram(image)
        
        # convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
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
        
        # generate features
        lbp = get_lbp(image)
        gabor = get_gabor_features(image)
        
        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([bgr, hsv, lab, [s_lines], [pp_lines], hist_o, hist_d, [g_corners],[l_corners], eod, hog, lbp, gabor])

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