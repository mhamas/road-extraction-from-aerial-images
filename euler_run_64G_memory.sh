#!/usr/bin/env bash

# bsub primer (see more at http://clusterwiki.ethz.ch/brutus/Using_the_batch_system)
# -n number of CPUs
# -R memory usage. "rusage[mem=2048]" takes 2GB memory PER CPU
# -W walltime
# -oo specifies output file


module load eth_proxy gcc/4.9.2 python/3.3.3 openblas/0.2.13_par
module load zlib

export BLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export LAPACK=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so
export ATLAS=/cluster/apps/openblas/0.2.13_par/x86_64/gcc_4.9.2/lib/libopenblas.so

export C_INCLUDE_PATH="$C_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/cluster/apps/python/3.3.3/x86_64/include/python3.3m/"

source ~/.venv/bin/activate

mkdir logs
cd src
bsub -n 8 -R "rusage[mem=8192]" -We 3:55 -oo "../logs/run_%J.out" \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/ext/lib" $HOME/ext/lib/ld-2.23.so $HOME/.venv/bin/python3 model_taivo_test.py
cd ..

deactivate # leave venv