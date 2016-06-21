#!/usr/bin/env bash

# bsub primer (see more at http://clusterwiki.ethz.ch/brutus/Using_the_batch_system)
# -n number of CPU cores
# -R memory usage. "rusage[mem=2048]" takes 2GB memory PER PROCESSOR
# -W walltime
# -oo specifies output file

mkdir logs
cd src
bsub LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/ext/lib" $HOME/ext/lib/ld-2.23.so $HOME/.venv/bin/python3 model_taivo_test.py \
-n 8 -R "rusage[mem=8192]" -W 3:00 -oo "../logs/run_%J.out"
cd ..