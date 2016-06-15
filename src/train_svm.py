import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import svm

import numpy as np

import patch_extraction_module as pem 
import data_loading_module as dlm
import constants as const

from sklearn.externals import joblib


def trainClassifier():
    print("Training SVM classifier (might take a while)")    
    
    train_data_filename = "../results/CNN_Output/training/raw/"
    #train_data_filename = "../data/training/images/"
    train_labels_filename = "../data/training/groundtruth/"
    
    num_images = 100
    
    # ground truth label images and CNN output
    labelsTrue = dlm.extract_label_images(train_labels_filename, num_images, const.IMG_PATCH_SIZE, const.IMG_PATCH_SIZE)
    labelsCNN  = dlm.read_image_array(train_data_filename, num_images, "raw_satImage_%.3d_patches")
    
    for i in range(0, len(labelsCNN)):
        labelsCNN[i] = resize(labelsCNN[i], (labelsCNN[i].shape[0] // const.IMG_PATCH_SIZE, labelsCNN[i].shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)        
    
    
    #plt.figure()
    #plt.imshow(labelsCNN[0])
    #plt.show()

    # extract patches and corresponding groundtruth center value
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
    
    clf = svm.SVC()
    clf.fit(X, y)

    # Evaluate model on training data
    y_new = np.squeeze(np.asarray([np.ravel(np.squeeze(np.asarray(p))[2, 2]) for p in patches]))
    error = np.sum((y - y_new) ** 2) / len(y_new)
    print("Training set accuracy: " + str(1 - error))
    
    y_new = clf.predict(X)
    error = np.sum((y - y_new) ** 2) / len(y_new)
    print("Postprocessing training set accuracy: " + str(1 - error))    
    
    return clf

def getSVMClassifier():
    fn = const.OBJECTS_PATH +"postprocessor.pkl"
    if not os.path.isfile(fn):        
        clf = trainClassifier()
        if not os.path.isdir(const.OBJECTS_PATH):
            os.makedirs(const.OBJECTS_PATH)
        joblib.dump(clf, const.OBJECTS_PATH + "postprocessor.pkl")
    else:
        clf = joblib.load(fn) 
        print("Loaded SVM")

    return clf
#%% Apply predictor on example
#img = labelsCNN[54]
#
#plt.figure()
#plt.imshow(img)
#plt.show()
#
#img_ptc = pem.img_crop(img, patch_size, border_size, 1, 0)
#
#output = clf.predict(np.asarray([np.ravel(np.squeeze(np.asarray(p))) for p in img_ptc]))
#
#outImg = np.reshape(output, img.shape, order=1)
#plt.figure()
#plt.imshow(outImg, cmap=plt.cm.gray, interpolation='nearest')
#plt.show()


