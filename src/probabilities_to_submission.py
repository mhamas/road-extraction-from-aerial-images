# Loads the probability images, applies postprocessing and then creates a submussion file

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import csv

from skimage.transform import resize

import constants as const
import trainDictionary as dict_train
import denoise_dictionary as dict_denoise
import denoise_low_rank as lowrank_denoise

D = dict_train.get_dictionary()

prob_fn = "../data/CNN_Output/Test/Probabilities/"

totalRawError = 0
totalDenoisedError = 0

verbose = False
num_images = 50
outputImages = []
for i in range(1, num_images+1):
    imageid = "raw_test_%d" % i
    image_filename = prob_fn + imageid + ".png"

    if os.path.isfile(image_filename):
        if verbose:
            print ('Loading ' + image_filename)
        img = mpimg.imread(image_filename)      
   
        # resize img to have 1 px for each 16 x 16 patch
        img = resize(img, (img.shape[0] // const.IMG_PATCH_SIZE, img.shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)        
        img2 = np.zeros(img.shape)
        img2[img >= 0.5] = 1 # this would be the prediction without post processing
        
		# APPLY POST PROCESSING 
        result = dict_denoise.denoiseImg(img, D)
        finalOutput = np.zeros(img.shape)
        finalOutput[result >= 0.5] = 1
        
# when using low rank approx, use 3 atoms instead of just 1
#        finalOutputLowRank = np.zeros(img.shape)
#        finalOutputLowRank[lowrank_denoise.denoiseImg(finalOutput) >= 0.5] = 1
        
#        plt.figure()
#        plt.imshow(finalOutput, cmap=plt.cm.gray, interpolation='nearest')
#        plt.show()      

		# END POST PROCESSING
		
        outputImages.append(finalOutput)
    else:
        print ('File ' + image_filename + ' does not exist')
        
        
#%% write out images into submission file
prefix_results = "../results/"
if not os.path.isdir(prefix_results):
    os.mkdir(prefix_results)

with open(prefix_results + "submission.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['id','prediction'])
    
    for i in range(1, 51):
        img = outputImages[i - 1].astype(np.int)
        # Saving to csv file for submission
        num_rows = img.shape[0]
        num_cols = img.shape[1]
        rows_out = np.empty((0,2))
        for x in range(0, num_rows):
            for y in range(0, num_cols):
                id = str(i).zfill(3) + "_" + str(const.IMG_PATCH_SIZE * x) + "_" + str(const.IMG_PATCH_SIZE * y)
                next_row = np.array([[id, str(img[y][x])]])
                rows_out = np.concatenate((rows_out, next_row))
        writer.writerows(rows_out)
    csvfile.close()    
            
    