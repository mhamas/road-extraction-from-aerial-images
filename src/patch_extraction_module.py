import os
import numpy as np
import scipy
import constants as const

def zero_center(patches):
    if os.path.isfile(const.PATCHES_MEAN_PATH + ".npy"):
        mean_patch = np.load(const.PATCHES_MEAN_PATH + ".npy")  
    else:
        mean_patch = np.mean(patches, axis = 0)
        np.save(const.PATCHES_MEAN_PATH, mean_patch)
        print('Mean patch saved to the disk.')
        
    return patches - mean_patch

def augment_image(img, out_ls, num_of_transformations):
    img2 = np.fliplr(img)
    # scipy.misc.imsave('02img.jpg', img2)

    out_ls.append(img)
    # scipy.misc.imsave('01img.jpg', img)

    if num_of_transformations > 0:
        tmp = np.rot90(img)
        out_ls.append(tmp)
        # scipy.misc.imsave('01rot90.jpg', tmp)
    if num_of_transformations > 1:
        tmp = np.rot90(np.rot90(img))
        out_ls.append(tmp)
        # scipy.misc.imsave('01rot180.jpg', tmp)
    if num_of_transformations > 2:
        tmp = np.rot90(np.rot90(np.rot90(img)))
        out_ls.append(tmp)
        # scipy.misc.imsave('01rot270.jpg', tmp)
    
    if num_of_transformations > 3:
        tmp = np.rot90(img2)
        out_ls.append(tmp)
        # scipy.misc.imsave('02rot90.jpg', tmp)
    if num_of_transformations > 4:
        tmp = np.rot90(np.rot90(img2))
        out_ls.append(tmp)
        # scipy.misc.imsave('02rot180.jpg', tmp)
    if num_of_transformations > 5:
        tmp = np.rot90(np.rot90(np.rot90(img2)))
        out_ls.append(tmp)
        # scipy.misc.imsave('02rot270.jpg', tmp)

def img_crop(im, patch_size, stride, num_of_transformations):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_single_channel = len(im.shape) < 3

    for i in range(0,imgheight - patch_size + 1, stride):
        for j in range(0,imgwidth - patch_size + 1, stride):
            if is_single_channel:
                # [1, patch_size, patch_size]
                im_patch = [im[j:j+patch_size, i:i+patch_size]]
            else:
                # [patch_size, patch_size, num_of_channels]
                im_patch = im[j:j+patch_size, i:i+patch_size, :]

            augment_image(im_patch, list_patches, num_of_transformations)

    return list_patches