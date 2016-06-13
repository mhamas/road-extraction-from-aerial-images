import os
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import svm

import time
import code
import numpy as np
import math
import scipy

import patch_extraction_module as pem 
import data_loading_module as dlm
import constants as const

train_data_filename = "../results/CNN_Output/training/raw/"
#train_data_filename = "../data/training/images/"
train_labels_filename = "../data/training/groundtruth/"

num_images = 3

# ground truth label images and CNN output
labelsTrue = dlm.extract_label_images(train_labels_filename, num_images, const.IMG_PATCH_SIZE, const.IMG_PATCH_SIZE)
labelsCNN  = dlm.read_image_array(train_data_filename, num_images, "raw_satImage_%.3d_patches")

for i in range(0, len(labelsCNN)):
    labelsCNN[i] = resize(labelsCNN[i], (labelsCNN[i].shape[0] // const.IMG_PATCH_SIZE, labelsCNN[i].shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)        


#plt.figure()
#plt.imshow(labelsCNN[0])
#plt.show()

#%%

# extract patches and corresponding groundtruth center value
patch_size = 1
border_size = 2
stride = 1

patches = []
labels = []
for i in range(0, num_images):
    patches.extend(pem.img_crop(labelsCNN[i], patch_size, border_size, 4, 1))
    labels.extend(pem.img_crop(labelsTrue[i], 1, 0, 4, 1))

    
#%%
X = np.asarray([np.ravel(np.squeeze(np.asarray(p))) for p in patches])
y = np.squeeze(np.asarray(labels))

clf = svm.SVC()
clf.fit(X, y)

# TODO: evaluate model...

