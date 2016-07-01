import numpy as np
import patch_extraction_module as pem 
import constants as const


def denoiseImg(img, clf):
    """Denoise an image of predictions img using the given SVM classifier clf."""
    
    img_ptc = pem.img_crop(img, 1, const.POSTPRO_SVM_PATCH_SIZE // 2, 1, 0)
    
    output = clf.predict(np.asarray([np.ravel(np.squeeze(np.asarray(p))) for p in img_ptc]))
    
    out_img = np.reshape(output, img.shape, order=1)
    return out_img
