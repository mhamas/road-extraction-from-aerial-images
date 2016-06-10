"""
CIL-Road-Segmentation
Matej Hamas, Taivo Pungas, Delio Vicini
Team: DataMinions

This script computes the output for the road segmentation task. If the necessary 
are not found cached on the disk, this script automatically trains them 
(which can take several hours, depending on machine configuration)


"""

# Train CNN
#TODO: Train CNN if needed, otherwise load model from disk

# Compute CNN predictions
#TODO: Generate predictions using CNN (save to image files)

# Apply post processing to CNN output
import probabilities_to_submission # generates the final CSV given the CNN probabilities (should be merged into postproc. module)



