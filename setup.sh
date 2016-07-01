#!/usr/bin/env bash

# Create data directories
mkdir -p data/test_set/downsampled
mkdir -p data/training/groundtruth/downsampled
mkdir -p data/training/images/downsampled

# Create output directories
mkdir -p results/CNN_Output/test/raw
mkdir -p results/CNN_Output/test/high_res_raw
mkdir -p results/CNN_Output/training/raw
mkdir -p results/CNN_Output/training/high_res_raw

mkdir -p results/CNN_Output_Baseline/test/raw
mkdir -p results/CNN_Output_Baseline/training/raw

# Create tmp directories to hold TensorFlow results
mkdir -p tmp
mkdir -p src/baseline/tmp
