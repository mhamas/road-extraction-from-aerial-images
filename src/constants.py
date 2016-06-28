# CONSTANTS THAT ARE COMMON TO MULTIPLE SOURCE FILES
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_PATCH_SIZE = 8
IMG_CONTEXT_SIZE = 64
IMG_BORDER_SIZE = int((IMG_CONTEXT_SIZE - IMG_PATCH_SIZE) / 2)
IMG_PATCH_STRIDE = 4

OBJECTS_PATH = "../objects/"
PATCHES_MEAN_PATH = OBJECTS_PATH + "patches_mean"

RESULTS_PATH = "../results"

# post processing
DICT_PATCH_SIZE = (5, 5)
LOW_RANK_TARGET = 3 # desired rank of output

POSTPRO_PATCH_SIZE = 16 # patch size we need to predict in the end
POSTPRO_SVM_PATCH_SIZE = 7 # use 7 to get best score with SVM
POSTPRO_CNN_PATCH_SIZE = 9 # use 7 to get best score with SVM