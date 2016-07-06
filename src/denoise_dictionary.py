import numpy as np
from sklearn.decomposition import SparseCoder
import constants as const


def denoiseImg(img, D):
    """Denoise prediction patches img using the dictionary D."""
    
    img_width = img.shape[0]
    img_height = img.shape[1]
    stride = 1  # denoise overlapping patches
    result = np.zeros((img_width, img_height))
    counts = np.zeros((img_width, img_height))
    
    coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=1, transform_alpha=1, transform_algorithm='omp')

    for i in range(0, img_height - const.DICT_PATCH_SIZE[1] + 1, stride):
        for j in range(0, img_width - const.DICT_PATCH_SIZE[0] + 1, stride):
            ptc = img[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]]
            x = coder.transform(ptc.reshape(1, -1))
            x = np.ravel(np.dot(x, D))
            x = x.reshape(const.DICT_PATCH_SIZE[0], const.DICT_PATCH_SIZE[1])
            result[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]] += x
            counts[j:j+const.DICT_PATCH_SIZE[0], i:i+const.DICT_PATCH_SIZE[1]] += np.ones(const.DICT_PATCH_SIZE)

    l = 0.0
    return (result + l * img) / (counts + l)
