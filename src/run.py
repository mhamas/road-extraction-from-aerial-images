"""
CIL-Road-Segmentation
Matej Hamas, Taivo Pungas, Delio Vicini
Team: DataMinions

This script computes the output for the road segmentation task. If the necessary 
are not found cached on the disk, this script automatically trains them 
(which can take several hours, depending on machine configuration)


"""
import glob
import postprocessing as pp
import model_taivo_test as cnn
import upsampling as upsampling

# Train CNN
cnn.main()

# Upsample predictions for both training and test set
UPSAMPLE = True
if UPSAMPLE:
    training_filenames = glob.glob("../results/CNN_Output/training/*/*.png")
    test_filenames = glob.glob("../results/CNN_Output/test/*/*.png")
    upsampling.upsample_training(training_filenames)
    upsampling.upsample_test(test_filenames)

# Apply post processing to CNN output
pp.generate_output()


