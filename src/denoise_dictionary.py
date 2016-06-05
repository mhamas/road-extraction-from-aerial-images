from time import time

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.image as mpimg

from skimage.transform import resize
from sklearn.decomposition import SparseCoder

import constants as const
import trainDictionary as dict_train

from PIL import Image

# Train dictionary and denoise image

LOAD_DICT_CACHE = True
CACHE_FILE_NAME = '../tmp/dict_cache.npy'
dict = []
if (LOAD_DICT_CACHE):
	if os.path.isfile(CACHE_FILE_NAME):
		dict = np.load(CACHE_FILE_NAME)
		print('Loaded dictionary from file')
		
if (dict == []):
	fn = "../data/training/groundtruth/"
	num_images = 50
	dict = dict_train.train_dictionary(fn, const.DICT_PATCH_SIZE, num_images)

if not os.path.exists('../tmp'):
	os.makedirs('../tmp')
np.save(CACHE_FILE_NAME, dict)

# dict_train.visualize_dictionary(dict, const.DICT_PATCH_SIZE)


# Load some result image and denoise it using our dictionary
imageid = "raw_satImage_005"
image_filename = "../data/CNN_Output/Training/Probabilities/" + imageid + ".png"
img = mpimg.imread(image_filename)

# resize img to have 1 px for each 16 x 16 patch
img = resize(img, (img.shape[0] // const.IMG_PATCH_SIZE, img.shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)
plt.figure()
plt.title('Input image')
plt.imshow(img, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')			 
plt.draw()

img2 = np.zeros(img.shape)
img2[img >= 0.5] = 1
plt.figure()
plt.title('Thresholded input')
plt.imshow(img2, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')	
plt.draw()

# denoise a single small patch	
# j = 5
# i = 1
# ptc = img[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]]
# plt.figure()
# plt.title('ptc')
# plt.imshow(ptc, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')			 
# plt.draw()

# solve OMP problem
# coder = SparseCoder(dictionary=dict, transform_n_nonzero_coefs=3,
					# transform_alpha=0.1, transform_algorithm='omp')
					
# x = coder.transform(ptc.reshape(1, -1))
# density = len(np.flatnonzero(x))
# x = np.ravel(np.dot(x, dict))
# x = x.reshape(5, 5)

# plt.figure()
# plt.title('ptc')
# plt.imshow(x, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')			 
# plt.draw()

# quit()

imgwidth = img.shape[0]
imgheight = img.shape[1]
stride = 1 # denoise overlapping patches
result = np.zeros((imgwidth, imgheight))
counts = np.zeros((imgwidth, imgheight))

coder = SparseCoder(dictionary=dict, transform_n_nonzero_coefs=3,
			transform_alpha=0.1, transform_algorithm='omp')
					
for i in range(0,imgheight - const.DICT_PATCH_SIZE[1] + 1, stride):
	for j in range(0,imgwidth - const.DICT_PATCH_SIZE[0] + 1, stride):
		ptc = img[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]]
					
		x = coder.transform(ptc.reshape(1, -1))
		density = len(np.flatnonzero(x))
		x = np.ravel(np.dot(x, dict))
		x = x.reshape(const.DICT_PATCH_SIZE[0], const.DICT_PATCH_SIZE[1])
		result[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]] += x
		counts[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]] += np.ones(const.DICT_PATCH_SIZE)
			
			
result /= counts
plt.figure()
plt.title('Denoised')
plt.imshow(result, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')	
plt.draw()

finalOutput = np.zeros((imgwidth, imgheight))
finalOutput[result >= 0.5] = 1
plt.figure()
plt.title('Thresholded output')
plt.imshow(finalOutput, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')	
plt.draw()

plt.show()

			