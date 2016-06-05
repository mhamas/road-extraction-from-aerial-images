import numpy as np
import constants as const

# simple low rank approximation of input image, can be useful for american style road layouts
def denoiseImg(img):
    U, s, V = np.linalg.svd(img, full_matrices=True)	
    rank = const.LOW_RANK_TARGET
    s[rank:] = 0
    S = np.zeros((U.shape[0], V.shape[0]))
    tmp = min(U.shape[0], V.shape[0])
    S[:tmp, :tmp] = np.diag(s)
    
    output = np.dot(U, np.dot(S, V))
    return output
