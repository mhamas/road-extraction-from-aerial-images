"""
CIL-Road-Segmentation
Matej Hamas, Taivo Pungas, Delio Vicini
Team: DataMinions

This script computes the output for the road segmentation task. If the necessary 
are not found cached on the disk, this script automatically trains them 
(which can take several hours, depending on machine configuration)


"""
import postprocessing as pp
import model_taivo_test as cnn

# Train CNN
cnn.main()

# Apply post processing to CNN output
pp.generate_output()


