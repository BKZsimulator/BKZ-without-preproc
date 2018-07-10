#!/bin/sh
# example for runing BKZ without preprocessing
for i in $(seq 1 1)
do
    nohup mpiexec -n 1 python main_pbkz_mpi.py -file lattice_input -bs 30 -cores 1 > test_output 2>&1 &
done
