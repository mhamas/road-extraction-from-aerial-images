"""
Provides methods to train a dictionary for dictionary-based denoising of predictions.
This code is used for the second baseline in the evaluation of our method.

"""

from time import time

import os

import matplotlib.pyplot as plt
import numpy as np

import data_loading_module as dlm
import constants as const

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d


def visualize_dictionary(V, patch_size):
    """ Visualizes the atoms in a learned dictionary V """
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary learned from groundtruth patches', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.draw()
    plt.show()


def train_dictionary(filename, patch_size, num_images):
    """ Trains a dictionary from the ground truth labels to denoise the CNN output """
    show_images = False
                    
    # 1 Load all the ground truth images => convert them into lowres label images
    print('Extracting reference patches...')
    t0 = time()
    labels = dlm.extract_label_images(filename, num_images, const.POSTPRO_PATCH_SIZE, const.POSTPRO_PATCH_SIZE)

    if show_images:
        plt.figure()
        plt.title('Image')
        plt.imshow(labels[3], vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')			 
        plt.draw()

    # 2 Extract patches from label images to learn dictionary
    data = np.asarray([extract_patches_2d(labels[i], patch_size) for i in range(num_images)])
    data = data.reshape(-1, data.shape[2], data.shape[3])

    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    print('done in %.2fs.' % (time() - t0))

    # 3 Train dictionary
    print('Learning the dictionary...')
    t0 = time()
    dico = MiniBatchDictionaryLearning(n_components=100, alpha=2, n_iter=2000)
    V = dico.fit(data).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)
    print('Trained on %d patches' % (len(data)))
    
    if show_images:
        visualize_dictionary(V, patch_size)
    return V


def get_dictionary():
    """ Returns a dictionary of prototypical patches of predictions. Caches the learned dictionary to disk """
    
    LOAD_DICT_CACHE = True
    CACHE_FILE_NAME = '../tmp/dict_cache.npy'
    loaded = False
    if LOAD_DICT_CACHE:
        if os.path.isfile(CACHE_FILE_NAME):
            D = np.load(CACHE_FILE_NAME)
            loaded = True
            print('Loaded dictionary from file')

    if not loaded:
        fn = "../data/training/groundtruth/"
        num_images = 100 
        D = train_dictionary(fn, const.DICT_PATCH_SIZE, num_images)
    
    if not os.path.exists('../tmp'):
        os.makedirs('../tmp')
    np.save(CACHE_FILE_NAME, D)    
    return D
