#!/usr/bin/env bash
mkdir results/euler_logs
scp -r -i ~/.ssh/id_rsa pungast@euler.ethz.ch:~/eth-cil-project/logs/*.out results/euler_logs