# mpiexec -n 4 python main_cleanbkz_mpi.py -file /home/shi/suite/sb_fpylll/bench/svpchallenge/svpchallengedim120seed0_LLL.txt -bs 90 -cores 4

import os, sys, pickle, numpy, platform
from copy import copy
from multiprocessing import Process
from time import sleep, clock, time
from random import randint
from math import log, sqrt, floor
from time import time
from fpylll import BKZ, LLL, GSO, IntegerMatrix, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZ1
from fpylll.tools.bkz_stats import BKZTreeTracer, dummy_tracer
from cleanbkz_mpi import BKZReduction as CLEANBKZ_MPI
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
machine = platform.node()


def main_cleanbkz_mpi_slave_block (bs, cores):

    # receiving parameters
    A = comm.recv(source=0, tag=1)
    kappa = comm.recv(source=0, tag=2)
    # max preprocessing block size 
    block_size = min(bs, A.nrows - kappa)

    # bkz object
    M = GSO.Mat(A)
    bkz_sub = CLEANBKZ_MPI(M, cores)
    bkz_sub.M.update_gso()
    params = BKZ.Param(block_size=block_size, max_loops=1,
                           min_success_probability=.01,
                           flags=BKZ.BOUNDED_LLL)
    # bkz_sub.randomize_block(kappa, kappa+block_size, density=block_size/4)
    # small block or larger block
    bkz_sub.svp_reduction_mpi(kappa, block_size, params, dummy_tracer)
    return


# called from ranks \ne 0
def main_cleanbkz_mpi_slave (bs , cores):

    while (1):
        sleep(.01)

        # end loop signal
        state = MPI.Status()
        end = comm.iprobe(source=0, tag=999, status=state)
        if(end):
            yesend = comm.irecv(source=0, tag=999)
            break
        
        # working signal
        state = MPI.Status()
        detect = comm.probe(source=0, tag=0, status=state)
        if (detect):
            signal = comm.recv(source=0, tag=0)
            if (signal==1):
                main_cleanbkz_mpi_slave_block (bs, cores)
            
    return


# only called by rank 0
def main_cleanbkz_mpi_master (filename, bs, cores):

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

    print r

    A = IntegerMatrix.from_matrix(L_Ainput_M.M.B, int_type="long")
    
    cleanbkz_mpi = CLEANBKZ_MPI(A, cores)
    #cleanbkz_mpi = BKZ2(A) 
    cleanbkz_mpi.lll_obj()
    cleanbkz_mpi.M.update_gso()
    r = [log(cleanbkz_mpi.M.get_r(i, i)) for i in range(0, cleanbkz_mpi.A.nrows)]
    print "# starting r "
    print r
    
    params = BKZ.Param (block_size=bs, max_loops=5000,
                            min_success_probability=.01,
                            flags=BKZ.BOUNDED_LLL, #|BKZ.DUMP_GSO,
                            dump_gso_filename="gso_output.file",
                            strategies = "default.json")
    cleanbkz_mpi(params=params, min_row=0)
    #print "# done. found sv", cleanbkz_mpi.M.B[0]

    # done send last end signal
    for i in range(1,size):
        comm.send(1, i, tag=999) # *** changed to send

    cleanbkz_mpi.M.update_gso()
    r2 = [log(cleanbkz_mpi.M.get_r(i, i)) for i in range(0, cleanbkz_mpi.A.nrows)]
    print cleanbkz_mpi.A[0]
    print "# ending r"
    print r2
    
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
    if rank == 0:
        main_cleanbkz_mpi_master (filename, bs, cores)
    else:
        main_cleanbkz_mpi_slave (bs, cores)
    return
        
if __name__ == '__main__':
    main()
    
