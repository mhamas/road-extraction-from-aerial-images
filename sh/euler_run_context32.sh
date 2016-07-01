#!/usr/bin/env bash

# bsub primer (see more at http://clusterwiki.ethz.ch/brutus/Using_the_batch_system)
# -n number of CPUs
# -R memory usage. "rusage[mem=2048]" takes 2GB memory PER CPU
# -W walltime
# -oo specifies output file


mkdir logs
cd src
bsub -n 8 -R "rusage[mem=8192]" -We 4:00 -oo "../logs/run_%J.out" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/ext/lib" $HOME/ext/lib/ld-2.23.so $HOME/.venv/bin/python3 model_large_context.py
cd ..