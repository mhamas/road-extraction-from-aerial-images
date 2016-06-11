import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv

from skimage.transform import resize

import constants as const
import trainDictionary as dict_train
import denoise_dictionary as dict_denoise
import denoise_low_rank as lowrank_denoise


# labels - 1 hot array of labels for each patch
# returned mask has size (# of patches) x (# of patches)
# (e.g. 25x25 for 16x16 patches and 400x400 image)
def prediction_to_mask(labels, width, height):
    mask = np.zeros([height, width])
    idx = 0
    for i in range(0,height):
        for j in range(0,width):
            mask[i][j] = 0 if labels[idx][0] > 0.5 else 1
            idx = idx + 1
    return mask

# Inverse of prediction_to_mask
def mask_to_prediction(mask):
    (height, width) = mask.shape;
    prediction = np.zeros([height * width, 2])
    idx = 0
    for i in range(0,height):
        for j in range(0,width):
            if mask[i][j] == 1:
                prediction[idx][1] = 1
            else:
                prediction[idx][0] = 1
            idx = idx + 1
    return prediction

# Basic 4 connectivity neighbour filtering
def set_to_zero_if_no_neighbours(mask):
    (height, width) = mask.shape
    for i in range(0, height):
        for j in range(0, width):
            if ((j == 0 and i == 0) or 
                (j == 0 and i == height - 1) or
                (j == width - 1 and i == 0) or
                (j == width - 1 and i == height - 1)):
                    continue                
            
            num_of_zero_neighbours = 0
            if i > 0 and mask[i - 1][j] == 0:
                num_of_zero_neighbours += 1
            if i < height - 1 and mask[i + 1][j] == 0:
                num_of_zero_neighbours += 1
            if j > 0 and mask[i][j - 1] == 0:
                num_of_zero_neighbours += 1
            if j < width - 1 and mask[i][j + 1] == 0:
                num_of_zero_neighbours += 1
                
            num_neighbors = 3 if (j == 0 or j == width - 1 or i == 0 or i == height - 1) else 4            

            num_of_one_neighbours = num_neighbors - num_of_zero_neighbours         
            if mask[i][j] == 1 and num_of_zero_neighbours >= 3:
                mask[i][j] = 0
            if mask[i][j] == 0 and num_of_one_neighbours >= 3:
                mask[i][j] = 1
    return mask

# prediction - 1 hot array of labels for each patch
def postprocess_prediction(prediction,  width, height):
    mask = prediction_to_mask(prediction, width, height);
    # scipy.misc.imsave('test_before.png', mask)
    mask = set_to_zero_if_no_neighbours(mask)
    # mask = scipy.signal.medfilt(mask, 3)
    # scipy.misc.imsave('test_after.png', mask)
    
    # dictionary based denoising
#    D = dict_train.get_dictionary()
#    mask = np.zeros(mask.shape)
#    mask[dict_denoise.denoiseImg(prediction[:][1], D) >= 0.5] = 1
    
    # simple low rank approximation
#    mask = np.zeros(mask.shape)
#    mask[lowrank_denoise.denoiseImg(prediction[:][1]) >= 0.5] = 1
    
    return mask_to_prediction(mask)
    
def apply_postprocessing(img, dictionary):
    # resize img to have 1 px for each 16 x 16 patch
    img = resize(img, (img.shape[0] // const.IMG_PATCH_SIZE, img.shape[1] // const.IMG_PATCH_SIZE), order=0, preserve_range=True)        
    img2 = np.zeros(img.shape)
    img2[img >= 0.5] = 1 # this would be the prediction without post processing
    
    
    finalOutput = set_to_zero_if_no_neighbours(img2)
    
#    result = dict_denoise.denoiseImg(img, dictionary)
#    finalOutput = np.zeros(img.shape)
#    finalOutput[result >= 0.5] = 1            
            
    # when using low rank approx, use 3 atoms instead of just 1
#    finalOutputLowRank = np.zeros(img.shape)
#    finalOutputLowRank[lowrank_denoise.denoiseImg(finalOutput) >= 0.5] = 1
#    
#    plt.figure()
#    plt.imshow(finalOutput, cmap=plt.cm.gray, interpolation='nearest')
#    plt.show()      
    return finalOutput
    
def create_submission_file(images):
    prefix_results = const.RESULTS_PATH + "/"
    if not os.path.isdir(prefix_results):
        os.mkdir(prefix_results)
    
    with open(prefix_results + "submission.csv", "w", newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id','prediction'])
        
        for i in range(1, len(images) + 1):
            img = images[i - 1].astype(np.int)
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
        
def save_images(images, imSizes):
    postpro_fn = const.RESULTS_PATH + "/postprocessing_output"
    fn = postpro_fn + "/test/"

    if not os.path.isdir(const.RESULTS_PATH):
        os.mkdir(const.RESULTS_PATH)
    if not os.path.isdir(postpro_fn):
        os.mkdir(postpro_fn)
    if not os.path.isdir(fn):
        os.mkdir(fn)
        
    for i in range(0, len(images)):
        scipy.misc.imsave(fn + ("satImage_%d" % i) + ".png" , resize(images[i],  imSizes[i], order=0, preserve_range=True))
    
def generate_output():  
    #prob_fn = "../data/CNN_Output/Test/Probabilities/"
    #prob_fn = "../results/CNN_Output/test/raw/"
    prob_fn = "../results/CNN_Output/test/high_res_raw/"  
    
    D = dict_train.get_dictionary()

    verbose = False
    num_images = 50
    outputImages = []
    imSizes = []
    for i in range(1, num_images+1):
    #    imageid = "raw_test_%d_patches" % i
        imageid = "raw_test_%d_pixels" % i
        image_filename = prob_fn + imageid + ".png"
    
        if os.path.isfile(image_filename):
            if verbose:
                print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename) 
            imSizes.append(img.shape)
            outputImages.append(apply_postprocessing(img, D) )
        else:
            print ('File ' + image_filename + ' does not exist')
            
    save_images(outputImages, imSizes)
    create_submission_file(outputImages)
    