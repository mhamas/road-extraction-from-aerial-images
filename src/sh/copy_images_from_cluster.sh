#!/usr/bin/env bash
mkdir results/euler/images
scp -r -i ~/.ssh/id_rsa pungast@euler.ethz.ch:~/eth-cil-project/results/predictions_* results/euler/images