import os
import matplotlib.image as mpimg
import numpy as np

import patch_extraction_module as pem
import constants as const

def extract_data(filename, num_images,  num_of_transformations = 6, patch_size = const.IMG_PATCH_SIZE, patch_stride = const.IMG_PATCH_STRIDE):
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (const.IMG_WIDTH / patch_size) * (const.IMG_HEIGHT / patch_size)
    print('Extracting patches...')
    img_patches = [pem.img_crop(imgs[i], patch_size, patch_stride, num_of_transformations) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    print(str(len(data)) + ' patches extracted.')

    patches = pem.zero_center(np.asarray(data))

    return patches


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        # Road
        return [0, 1]
    else:
        # Non-road
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images, num_of_transformations = 6, patch_size = const.IMG_PATCH_SIZE, patch_stride = const.IMG_PATCH_STRIDE):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    print('Extracting patches...')
    gt_patches = [pem.img_crop(gt_imgs[i], patch_size, patch_stride, num_of_transformations) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    print(str(len(data)) + ' label patches extracted.')

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)
    
# converts 1-hot pixel-wise labels to a lowres image of patch labels
def pixel_to_patch_labels(im, patch_size, stride):
    imgwidth = im.shape[0]
    imgheight = im.shape[1]

    outW = 0
    outH = 0
    for i in range(0,imgheight - patch_size + 1, stride):
        outH = outH + 1
    for j in range(0,imgwidth - patch_size + 1, stride):
        outW = outW + 1
       
    output = np.zeros((outW, outH));
    idxI = 0
    for i in range(0,imgheight - patch_size + 1, stride):
        idxJ = 0
        for j in range(0,imgwidth - patch_size + 1, stride):
            im_patch = [im[j:j+patch_size, i:i+patch_size]]
            output[idxJ, idxI] = value_to_class(np.mean(im_patch))[1]
            idxJ = idxJ + 1

        idxI = idxI + 1
        
    return output
    
# extract labels from ground truth as label images
def extract_label_images(filename, num_images, patch_size = const.IMG_PATCH_SIZE, patch_stride = const.IMG_PATCH_STRIDE):
    """Extract the labels into label images"""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    print('Extracting patches...')
    gt_patches = [pixel_to_patch_labels(gt_imgs[i], patch_size, patch_stride) for i in range(num_images)]
    
    return gt_patches
    