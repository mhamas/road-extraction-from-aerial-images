#!/usr/bin/env bash

# Uses https://github.com/andreafabrizi/Dropbox-Uploader
~/Dropbox-Uploader/dropbox_uploader.sh upload ~/eth-cil-project/results/predictions_training/raw* /predictions_training
~/Dropbox-Uploader/dropbox_uploader.sh upload ~/eth-cil-project/results/predictions_test/raw* /predictions_test
