# mpiexec -n 4 python main_pruning.py -file /home/shi/suite/sb_fpylll/bench/svpchallenge/svpchallengedim120seed0_LLL.txt -bs 90 -cores 1

import os, sys, pickle, numpy, platform
from copy import copy
from multiprocessing import Process
from time import sleep, clock, time
from random import randint
from math import log, sqrt, floor
from time import time
from fpylll import BKZ, LLL, GSO, IntegerMatrix, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZ1
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.tools.bkz_stats import BKZTreeTracer, dummy_tracer
from pbkz_mpi import BKZReduction as PBKZ_MPI
from fpylll.fplll.pruner import Pruning
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
machine = platform.node()

NPS = 60*[2.**29] + 5 * [2.**27] + 5 * [2.**26] + 1000 * [2.**25]

# only called by rank 0
def main_pruning (filename, bs, cores):

    try:
        with open(filename, "rb") as f:
            mat = pickle.load(f)
            #print "len(mat)", len(mat)
            #if (len(mat) > 1):
            #   mat = mat[0]
        if isinstance(mat, IntegerMatrix):
            Ainput = mat
        else:
            Ainput = IntegerMatrix.from_matrix(mat)
    except:
        Ainput = IntegerMatrix.from_file(filename)

    Ainput_M = GSO.Mat(Ainput, float_type='double')
    Ainput_M.update_gso()
    r = [Ainput_M.get_r(i, i) for i in range(0, Ainput.nrows)]
    L_Ainput_M = LLL.Reduction(Ainput_M)
    L_Ainput_M()
    #print r

    
    A = IntegerMatrix.from_matrix(L_Ainput_M.M.B, int_type="long")
    M = GSO.Mat(A, float_type="double")
    bkzobj = BKZ2(M)
    bkzobj.M.update_gso()
    block_size = bs
    r = [M.get_r(i, i) for i in range(0, block_size)]
    radius = r[0] * 0.99
    preproc_cost = 5000**(rank + 1)
    
    pr0 = Pruning.run(radius, NPS[block_size] * preproc_cost, [r], 0.1,
                        metric="probability", float_type="double",
                        flags=Pruning.GRADIENT|Pruning.NELDER_MEAD)
    
    print pr0.coefficients
    """
    pruning = prune(radius, NPS[block_size] * preproc_cost, [r], 0.01,
                        metric="probability", float_type="double",
                        flags=Pruning.GRADIENT|Pruning.NELDER_MEAD)
    cost = sum(pruning.detailed_cost) / NPS[block_size]
    print "# [rank %d] cost %.1f, precost %.1f " % (rank, cost, preproc_cost)
    """

    
    pr0_linear = pr0.LinearPruningParams(block_size, block_size-2)
    print pr0_linear.coefficients



    
    return


def parse_options(comm):
    parser = argparse.ArgumentParser()
    parser.add_argument('-file')
    parser.add_argument('-bs')
    parser.add_argument('-cores')
    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = comm.bcast(args, root=0)
    if args is None:
        exit(0)
    return args


def main():
    
    # parse argument
    args = parse_options(comm)
    if rank == 0:
        print "###################################### "
        print ("# [Args] MPI: %d" % size)
        print ("# [Args] file: %s" % args.file)
        print ("# [Args] block_size: %s" % args.bs)
        print ("# [Args] cores per MPI: %s" % args.cores)
        print "###################################### "
        
    # start process
    bs = int(args.bs)
    cores = int(args.cores)
    filename = args.file
    main_pruning (filename, bs, cores)
    
    return
        
if __name__ == '__main__':
    main()
    
