"""

This module provides helper functions to extract patches from images. This 
includes data augmentation and mirror boundary conditions

"""

import os
import numpy as np
import constants as const


def zero_center(patches):
    """ Zero centers patch data and caches their mean value to disk """    

    if os.path.isfile(const.PATCHES_MEAN_PATH + ".npy"):
        mean_patch = np.load(const.PATCHES_MEAN_PATH + ".npy")  
    else:
        if not os.path.isdir(const.OBJECTS_PATH):            
            os.makedirs(const.OBJECTS_PATH)
        print("Computing mean patch")
        mean_patch = np.mean(patches, axis = 0)
        print("Mean computed")
        np.save(const.PATCHES_MEAN_PATH, mean_patch)
        print("Mean patch saved to the disk.")

    return patches - mean_patch
    

def augment_image(img, out_ls, num_of_transformations):
    """ Augments the input image img by a number of transformations (rotations by 90° and flips). 
        out_ls --- list of output images
        num_of_transformations --- number of transformations to compute """
        
    img2 = np.fliplr(img)

    out_ls.append(img)

    if num_of_transformations > 0:
        tmp = np.rot90(img)
        out_ls.append(tmp)
    if num_of_transformations > 1:
        tmp = np.rot90(np.rot90(img))
        out_ls.append(tmp)
    if num_of_transformations > 2:
        tmp = np.rot90(np.rot90(np.rot90(img)))
        out_ls.append(tmp)
    
    if num_of_transformations > 3:
        tmp = np.rot90(img2)
        out_ls.append(tmp)
    if num_of_transformations > 4:
        tmp = np.rot90(np.rot90(img2))
        out_ls.append(tmp)
    if num_of_transformations > 5:
        tmp = np.rot90(np.rot90(np.rot90(img2)))
        out_ls.append(tmp)
    if num_of_transformations > 6:
        out_ls.append(img2)
        

def mirror_border(img, border_size):
    """ Pads an input image img with a border of size border_size using a mirror boundary condition """
    
    if (len(img.shape) < 3):
        # Binary image
        res_img = np.zeros((img.shape[0] + 2 * border_size, img.shape[1] + 2 * border_size))
    else:
        # 3 channel image
        res_img = np.zeros((img.shape[0] + 2 * border_size, img.shape[1] + 2 * border_size, 3))
    for i in range(border_size):
        res_img[border_size : res_img.shape[0] - border_size, border_size - 1 - i] = img[:, i]                                     # left columns
        res_img[border_size : res_img.shape[0] - border_size, res_img.shape[1] - border_size + i] = img[:, img.shape[1] - 1 - i]   # right columns
        res_img[border_size - 1 - i, border_size : res_img.shape[1] - border_size] = img[i, :]                                     # top rows
        res_img[res_img.shape[0] - border_size + i, border_size : res_img.shape[1] - border_size] = img[img.shape[0] - 1 - i, :]   # bottom rows
    res_img[border_size : res_img.shape[0] - border_size, border_size : res_img.shape[1] - border_size] = np.copy(img)
    # Corners
    res_img[0 : border_size, 0 : border_size] = \
        np.fliplr(np.flipud(img[0 : border_size, 0 : border_size]))
    res_img[0 : border_size, res_img.shape[1] - border_size : res_img.shape[1]] = \
        np.fliplr(np.flipud(img[0 : border_size, img.shape[1] - border_size : img.shape[1]]))
    res_img[res_img.shape[0] - border_size : res_img.shape[0], 0 : border_size] = \
        np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], 0 : border_size]))
    res_img[res_img.shape[0] - border_size : res_img.shape[0], res_img.shape[1] - border_size : res_img.shape[1]] = \
        np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], img.shape[1] - border_size : img.shape[1]])) 

    return res_img


def img_crop(im, patch_size, border_size, stride, num_of_transformations):
    """ Extracts patches of size patch_size and stride stride from an image img """
    
    context_size = patch_size + 2 * border_size    
    im = mirror_border(im, border_size)
    
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_single_channel = len(im.shape) < 3

    for i in range(0,imgheight - context_size + 1, stride):
        for j in range(0,imgwidth - context_size + 1, stride):
            if is_single_channel:
                # [1, patch_size, patch_size]
                im_patch = [im[j:j+context_size, i:i+context_size]]
            else:
                # [patch_size, patch_size, num_of_channels]
                im_patch = im[j:j+context_size, i:i+context_size, :]

            augment_image(im_patch, list_patches, num_of_transformations)

    return list_patches
    

def input_img_crop(im, patch_size, border_size, stride, num_of_transformations):
    """ Crops an input image. Direct alias of img_crop """
    
    return img_crop(im, patch_size, border_size, stride, num_of_transformations)
    

def label_img_crop(im, patch_size, stride, num_of_transformations):
    """ Crops a label image into patches """
    
    return img_crop(im, patch_size, 0, stride, num_of_transformations)



