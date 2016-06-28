# Simple script which rotates all training and groundtruth images by 45° to help with rotational invariance

import os
import matplotlib.image as mpimg
import numpy as np

import patch_extraction_module as pem
import constants as const
        
from skimage.transform import rotate        
        
num_images = 100

# sat images
filename = "../data/training/images/"
imgBaseName = "satImage_%.3d"

# groundtruth labels
#filename = "../data/training/groundtruth/"
#imgBaseName = "satImage_%.3d"

for i in range(1, num_images+1):
    imageid = imgBaseName % i
    image_filename = filename + imageid + ".png"
    
    imageid = imgBaseName % (i + num_images)
    image_output_filename = filename + imageid + ".png"

    if os.path.isfile(image_filename):
        print ('Loading ' + image_filename)
        img = mpimg.imread(image_filename)
        img2 = rotate(img, 45, mode="reflect")
#        mpimg.imsave(image_output_filename, 1-img2, cmap="Greys")
        mpimg.imsave(image_output_filename, img2)
    else:
        print ('File ' + image_filename + ' does not exist')

