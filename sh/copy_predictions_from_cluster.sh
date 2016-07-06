#!/usr/bin/env bash
mkdir results/euler
scp -r -i ~/.ssh/id_rsa pungast@euler.ethz.ch:~/eth-cil-project/results/submission.csv results/euler/
scp -r -i ~/.ssh/id_rsa pungast@euler.ethz.ch:~/eth-cil-project/results/baseline_submission.csv results/euler/