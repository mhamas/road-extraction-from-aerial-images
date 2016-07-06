"""
Provides methods to train a SVM for SVM-based post-processing of CNN predictions.
(training takes 1-2 hours, depending on machine configuration)
"""

import os
import time as time

import numpy as np

from skimage.transform import resize
from sklearn import svm
from sklearn.externals import joblib

import patch_extraction_module as pem 
import data_loading_module as dlm
import constants as const


def trainClassifier():
    """ Trains an SVM classifier to post-process the CNN output """
    
    print("Training SVM classifier (might take a while)")    
    t = time.time()
   
    train_data_filename = "../results/CNN_Output/training/raw/"
    train_labels_filename = "../data/training/groundtruth/"
    
    num_images = 100
    
    # ground truth label images and CNN output
    labelsTrue = dlm.extract_label_images(train_labels_filename, num_images, const.POSTPRO_PATCH_SIZE, const.POSTPRO_PATCH_SIZE)
    labelsCNN  = dlm.read_image_array(train_data_filename, num_images, "raw_satImage_%.3d_patches")
    
    for i in range(0, len(labelsCNN)):
        labelsCNN[i] = resize(labelsCNN[i], (labelsCNN[i].shape[0] // const.POSTPRO_PATCH_SIZE,
                                             labelsCNN[i].shape[1] // const.POSTPRO_PATCH_SIZE),
                              order=0, preserve_range=True)
        
    elapsed = time.time() - t
    print("Loading training data took: " + str(elapsed) + " s")

    # extract patches and corresponding groundtruth center value
    t = time.time()
    patch_size = 1
    border_size = const.POSTPRO_SVM_PATCH_SIZE // 2

    stride = 1
    nTransforms = 5
    
    patches = []
    labels = []
    for i in range(0, num_images):
        patches.extend(pem.img_crop(labelsCNN[i], patch_size, border_size, stride, nTransforms))
        labels.extend(pem.img_crop(labelsTrue[i], 1, 0, stride, nTransforms))

    X = np.asarray([np.ravel(np.squeeze(np.asarray(p))) for p in patches])
    y = np.squeeze(np.asarray(labels))
    
    elapsed = time.time() - t
    print("Extracting patches from training data took: " + str(elapsed) + " s")
    print("Training set size: " + str(X.shape))
    print("Fitting SVM...")
    t = time.time()
    
    classifier = svm.SVC()    
    classifier.fit(X, y)
    
    elapsed = time.time() - t
    print("Training SVM took " + str(elapsed) + " s")

    # Evaluate model on training data
    y_new = np.squeeze(np.asarray([np.ravel(np.squeeze(
        np.asarray(p))[const.POSTPRO_SVM_PATCH_SIZE // 2, const.POSTPRO_SVM_PATCH_SIZE // 2])for p in patches]))
    error = np.sum((y - y_new) ** 2) / len(y_new)
    print("Training set accuracy: " + str(1 - error))
    
    y_new = classifier.predict(X)
    error = np.sum((y - y_new) ** 2) / len(y_new)
    print("Postprocessing training set accuracy: " + str(1 - error))    
    
    return classifier


def getSVMClassifier():
    """ Returns a SVM classifier to post-process the predictions. Caches the learned SVM to disk """

    fn = const.OBJECTS_PATH + "postprocessor.pkl"
    if not os.path.isfile(fn):        
        clf = trainClassifier()
        if not os.path.isdir(const.OBJECTS_PATH):
            os.makedirs(const.OBJECTS_PATH)
        joblib.dump(clf, const.OBJECTS_PATH + "postprocessor.pkl")
    else:
        clf = joblib.load(fn) 
        print("Loaded SVM")

    return clf
