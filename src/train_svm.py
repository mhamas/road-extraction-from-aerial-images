import os
import time as time

import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize
from sklearn import svm
from sklearn.externals import joblib
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import patch_extraction_module as pem 
import data_loading_module as dlm
import constants as const

# POSTPROCESSING: Trains an SVM classifier to postprocess the CNN output

def trainClassifier():
    print("Training SVM classifier (might take a while)")    
    t = time.time()
   
    train_data_filename = "../results/CNN_Output/training/raw/"
    #train_data_filename = "../data/training/images/"
    train_labels_filename = "../data/training/groundtruth/"
    
    num_images = 10
    
    # ground truth label images and CNN output
    labelsTrue = dlm.extract_label_images(train_labels_filename, num_images, const.IMG_PATCH_SIZE, const.IMG_PATCH_SIZE)
    labelsCNN  = dlm.read_image_array(train_data_filename, num_images, "raw_satImage_%.3d_patches")
    
    for i in range(0, len(labelsCNN)):
        labelsCNN[i] = resize(labelsCNN[i], (labelsCNN[i].shape[0] // const.IMG_PATCH_SIZE, labelsCNN[i].shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)        
        
    elapsed = time.time() - t
    print("Loading training data took: " + str(elapsed) + " s")
    #plt.figure()
    #plt.imshow(labelsCNN[0])
    #plt.show()

    # extract patches and corresponding groundtruth center value
    t = time.time()
    patch_size = 1
    border_size = const.POSTPRO_SVM_PATCH_SIZE // 2
    stride = 2
    nTransforms = 3
    
# OLD SETTINGS
#    stride = 1
#    nTransforms = 5
    
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
    
    svc_model = svm.SVC()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('svm', svc_model)])
    
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 50

    classifier.fit(X, y)
    elapsed = time.time() - t
    print("Training SVM took " + str(elapsed) + " s")

    # Evaluate model on training data
    y_new = np.squeeze(np.asarray([np.ravel(np.squeeze(np.asarray(p))[const.POSTPRO_SVM_PATCH_SIZE // 2, const.POSTPRO_SVM_PATCH_SIZE // 2]) for p in patches]))
    error = np.sum((y - y_new) ** 2) / len(y_new)
    print("Training set accuracy: " + str(1 - error))
    
    y_new = classifier.predict(X)
    error = np.sum((y - y_new) ** 2) / len(y_new)
    print("Postprocessing training set accuracy: " + str(1 - error))    
    
    return classifier

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


