import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

from skimage.transform import resize

import constants as const
import trainDictionary as dict_train
import denoise_dictionary as dict_denoise
import denoise_low_rank as lowrank_denoise

import data_loading_module as dlm

def evaluate_error(img, ref):
    return np.sqrt(np.sum((img - ref) ** 2)) / np.prod(img.shape)
    

#%% Show dictionary denoising

plt.close('all')

# Train dictionary and denoise image
D = dict_train.get_dictionary()

# Load some result image and denoise it using our dictionary
imageid = "satImage_035"
imageid = "satImage_041"
image_filename = "../data/CNN_Output/Training/Probabilities/raw_" + imageid + ".png"
ref_filename = "../data/training/groundtruth/" + imageid + ".png"

img = mpimg.imread(image_filename)

# resize img to have 1 px for each 16 x 16 patch
img = resize(img, (img.shape[0] // const.IMG_PATCH_SIZE, img.shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)
plt.figure()
plt.title('Input image')
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')			 

img2 = np.zeros(img.shape)
img2[img >= 0.5] = 1
plt.figure()
plt.title('Thresholded input')
plt.imshow(img2, cmap=plt.cm.gray, interpolation='nearest')	

result = dict_denoise.denoiseImg(img, D)

plt.figure()
plt.title('Denoised, rank ' + str(np.linalg.matrix_rank(result)))
plt.imshow(result, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')	

finalOutput = np.zeros(img.shape)
finalOutput[result >= 0.5] = 1
plt.figure()
plt.title('Thresholded output, rank ' + str(np.linalg.matrix_rank(finalOutput)))
plt.imshow(finalOutput, cmap=plt.cm.gray, interpolation='nearest')

finalOutputLowRank = np.zeros(img.shape)
finalOutputLowRank[lowrank_denoise.denoiseImg(finalOutput) >= 0.5] = 1
plt.figure()
plt.title('Lowrank output, rank ' + str(np.linalg.matrix_rank(finalOutputLowRank)))
plt.imshow(finalOutputLowRank, cmap=plt.cm.gray, interpolation='nearest')


ref = mpimg.imread(ref_filename)     
ref = dlm.pixel_to_patch_labels(ref, const.IMG_PATCH_SIZE, const.IMG_PATCH_SIZE)
plt.figure()
plt.title('Reference, rank ' + str(np.linalg.matrix_rank(ref)))
plt.imshow(ref, cmap=plt.cm.gray, interpolation='nearest')	

plt.show()

#%% Evaluate dict denoising quality
D = dict_train.get_dictionary()

prob_fn = "../data/CNN_Output/Training/Probabilities/"
ref_fn = "../data/training/groundtruth/"

totalRawError = 0
totalDenoisedError = 0

verbose = False
num_images = 100
for i in range(1, num_images+1):
    imageid = "satImage_%.3d" % i
    image_filename = prob_fn + "raw_" + imageid + ".png"
    ref_filename = ref_fn + imageid + ".png"

    if os.path.isfile(image_filename) and os.path.isfile(ref_filename):
        if verbose:
            print ('Loading ' + image_filename)
        img = mpimg.imread(image_filename)      
        if verbose:
            print ('Loading ' + ref_filename)
        ref = mpimg.imread(ref_filename)     
        ref = dlm.pixel_to_patch_labels(ref, const.IMG_PATCH_SIZE, const.IMG_PATCH_SIZE)

        # resize img to have 1 px for each 16 x 16 patch
        img = resize(img, (img.shape[0] // const.IMG_PATCH_SIZE, img.shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)        
        img2 = np.zeros(img.shape)
        img2[img >= 0.5] = 1 # this would be the prediction without post processing
        
        result = dict_denoise.denoiseImg(img, D)
        finalOutput = np.zeros(img.shape)
        finalOutput[result >= 0.5] = 1
        
# when using low rank approx, use 3 atoms instead of just 1
#        finalOutputLowRank = np.zeros(img.shape)
#        finalOutputLowRank[lowrank_denoise.denoiseImg(finalOutput) >= 0.5] = 1
        
        rawError = evaluate_error(img2, ref)
        denoisedError = evaluate_error(finalOutput, ref)
        
        totalRawError += rawError
        totalDenoisedError += denoisedError
        if verbose:
            print(rawError)
            print(denoisedError)
        
    else:
        if verbose:
            if not os.path.isfile(image_filename):
                print ('File ' + image_filename + ' does not exist')
            if not os.path.isfile(ref_filename):
                print ('File ' + ref_filename + ' does not exist')

print("Avg. raw error: " + str(totalRawError / num_images))
print("Avg. denoised error: " +  str(totalDenoisedError / num_images))