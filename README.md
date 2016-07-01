# Road Extraction from Aerial Images
[Delio Vicini](https://github.com/dvicini), [Matej Hamas](https://github.com/mato93), [Taivo Pungas](https://github.com/taivop) (Department of Computer Science, ETH Zurich, Switzerland)

The code in this repository trains a convolutional neural network and adds a post-processing layer, for the task of detecting roads in satellite images. See project report [here](docs/report/main.pdf).

## Dependencies
* Python
 * scipy
 * numpy
 * scikit-learn
 * tensorflow
 * matplotlib
 * Pillow
 * skimage

## Setup
* Copy the data so the `data/` folder contains the following:
```
test_set/
  |-- downsampled/
  |-- test_1.png
  |-- test_2.png
  |-- ...
  |-- test_50.png
  
training/
  |-- groundtruth/
  |   |-- downsampled/
  |   |-- satImage001.png
  |   |-- satImage002.png
  |   |-- ...
  |   |-- satImage100.png
  |
  |-- images/
      |-- downsampled/
      |-- satImage001.png
      |-- satImage002.png
      |-- ...
      |-- satImage100.png
```
* Run `sh setup.sh` while in the project root folder to set up the necessary file structures (assumes a Bash shell).


## Running
**Warning:** running the code as-is requires around 100GB of memory.

* Run `python run.py` while in the `src` folder to train the CNN, apply post-processing and generate predictions.
* All results will be in the `results` folder, under `CNN_Output/`, `postprocessing_output/` and in `submission.csv`.

## Running the baseline
* Run `python baseline/model_baseline.py` while in the `src` folder.
* All results will be in the `results` folder, under `CNN_Output_Baseline/` and in `baseline_submission.csv`.

The project has been tested with and is guaranteed to run on Python 3.5.0.
