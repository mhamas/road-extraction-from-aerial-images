import numpy as np
import patch_extraction_module as pem 
import constants as const

def denoiseImg(img, clf):
    """ Denoises an image of predictions img using the given SVM classifier clf """
    
    img_ptc = pem.img_crop(img, 1, const.POSTPRO_SVM_PATCH_SIZE // 2, 1, 0)
    
    output = clf.predict(np.asarray([np.ravel(np.squeeze(np.asarray(p))) for p in img_ptc]))
    
    outImg = np.reshape(output, img.shape, order=1)
    return outImg