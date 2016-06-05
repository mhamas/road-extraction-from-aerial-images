import numpy as np
from sklearn.decomposition import SparseCoder
import constants as const

def denoiseImg(img, D):
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    stride = 1 # denoise overlapping patches
    result = np.zeros((imgwidth, imgheight))
    counts = np.zeros((imgwidth, imgheight))
    
    coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=2,
    			transform_alpha=1, transform_algorithm='omp')
    					
    for i in range(0,imgheight - const.DICT_PATCH_SIZE[1] + 1, stride):
    	for j in range(0,imgwidth - const.DICT_PATCH_SIZE[0] + 1, stride):
    		ptc = img[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]]					
    		x = coder.transform(ptc.reshape(1, -1))
    		x = np.ravel(np.dot(x, D))
    		x = x.reshape(const.DICT_PATCH_SIZE[0], const.DICT_PATCH_SIZE[1])
    		result[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]] += x
    		counts[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]] += np.ones(const.DICT_PATCH_SIZE)			
    			
    l = 0.0
    return (result + l * img) / (counts + l)
