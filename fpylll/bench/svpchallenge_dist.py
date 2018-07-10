# this use naive MPI and each of threading

# mpiexec -n 4 python svpchallenge_dist.py -file /home/shi/suite/sb_fpylll/bench/svpchallenge/svpchallengedim120seed0_LLL.txt -bs_diff 30 -cores 4

# system
import os, sys, pickle, numpy, platform
from copy import copy
from multiprocessing import Process, Queue
from time import sleep, clock, time
from random import randint
from math import log, sqrt, floor
# fpylll
from fpylll import IntegerMatrix, LLL, GSO
from fpylll import BKZ as fplll_bkz
from fpylll import Enumeration, EnumerationError
from fpylll.algorithms.bkz2_otf_subsol import BKZReduction
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning
from fpylll.util import gaussian_heuristic
import argparse
# mpi
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
machine = platform.node()
NPS = 2**24


###########################################
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file')
    parser.add_argument('-bs_diff')
    parser.add_argument('-cores')
    args = None
    args = parser.parse_args()
    if args is None:
        exit(0)
    return args




###########################################
def copy_to_IntegerMatrix_long(A):
    n = A.nrows
    AA = IntegerMatrix(n, n, int_type="long")
    for i in xrange(n):
        for j in xrange(n):
            AA[i, j] = A[i, j]
    return AA


###########################################
def insert_in_IntegerMatrix(A, v):
    n = A.nrows
    AA = IntegerMatrix(n + 1, n, int_type="long")
    for j in xrange(n):
        AA[0, j] = v[j]
        for i in xrange(n):
            AA[i + 1, j] = A[i, j]
    LLL.reduction(AA)
    for j in xrange(n):
        for i in xrange(n):
            A[i, j] = AA[i + 1, j]
    del AA

    
###########################################
def print_basis_stats(M, n):
    r = [M.get_r(i, i) for i in range(n)]
    gh = gaussian_heuristic(r)
    lhvr = log(gaussian_heuristic(r[:n/2])) - log(gaussian_heuristic(r[n/2:]))
    print "lhrv = %.4f, r[i]/gh"%lhvr,
    for i in range(20):
        print "%.3f"%(r[i]/gh), 
    print
    return


###########################################
def insert_sub_solutions(bkz, sub_solutions):
    M = bkz.M
    l = len(sub_solutions)
    n = M.d
    for (a, vector) in sub_solutions:
        M.create_row()
        if len(vector)==0:      # No subsolution at this index. Leaving a 0 vector
            continue 
        with M.row_ops(M.d-1, M.d):
            for i in range(n):                    
                M.row_addmul(M.d-1, i, vector[i])    
    for k in reversed(range(l)):
        M.move_row(M.d-1, k)
    bkz.lll_obj()
    for i in range(l):
        M.remove_last_row()
    return


###########################################
def svp_improve_trial_enum(bkz, preproc_cost, radius):
    verbose = 0
    n = bkz.A.nrows
    r = [bkz.M.get_r(i, i) for i in range(0, n)]       
    gh = gaussian_heuristic(r)
    PRUNE_START = time()
    pruning = prune(radius, NPS * preproc_cost, [r], 10, 
                    metric="solutions", float_type="dd",
                    flags=Pruning.GRADIENT|Pruning.NELDER_MEAD)
    PRUNE_TIME = time() - PRUNE_START 
    ENUM_START = time()
    enum_obj = Enumeration(bkz.M, sub_solutions=True)
    success = False
    try:        
        enum_obj.enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        success = True
    except EnumerationError:
        pass
    ENUM_TIME = time() - ENUM_START
    if (verbose):
        print ("# [Prune] time %.4f"%PRUNE_TIME)
        print ("# [Enum]  (Expecting %.5f solutions)"%(pruning.expectation)),
        print (", TIME = %.2f"%ENUM_TIME)
    """
    for (a, b) in enum_obj.sub_solutions[:20]:
        print "%.3f"%abs(a/gh),
    print 
    """
    #  insert_sub_solutions(bkz, enum_obj.sub_solutions)     ???
    insert_sub_solutions(bkz, enum_obj.sub_solutions[:n/4])
    return success


###########################################
def svp_improve_trial(A, bs, queue, filename):
    n = A.nrows
    bkz = BKZReduction(A)
    BKZ_START = time()
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(n)]
    gh = gaussian_heuristic(r)
    for lbs in range(30, bs - 10, 2) + [bs]:
        params = fplll_bkz.Param(block_size=lbs, max_loops=1,
                                 min_success_probability=.01)
        bkz(params=params)
        bkz.lll_obj()

    r = [bkz.M.get_r(i, i) for i in range(n)]

    BKZ_TIME = time() - BKZ_START
    #print "#  BKZ-%d, proc %s, time %.2f" %(bs, os.getpid(), BKZ_TIME)  
    ENUM_START = time()
    success = svp_improve_trial_enum(bkz, BKZ_TIME, r[0]*.99)
    ENUM_TIME = time() - ENUM_START
    #print "#  SVP-%d, proc %s, time %.2f, succ. %s" %(bs, os.getpid(), ENUM_TIME, success)
    length = bkz.M.get_r(0, 0)
    print "#   [rank %d] SVP-%d trial, proc %d, time %.2f, norm %d" %(rank, bs, os.getpid(), ENUM_TIME + BKZ_TIME, length)
    
    #queue.put(A)
    #queue.put(success)
    pickle.dump((A, success), open(filename, 'wb'))
    
    return success


class SVPool:
    def __init__(self, max_len, copies=1):
        self.max_len = max_len
        self.copies = copies
        self.data = []

    def push(self, v):
        norm = sum([x*x for x in v])
        for i in range(self.copies):
            self.data += [copy((norm, v))]
        if len(self.data)> self.max_len:
            self.data.sort()

    def pop(self):
        l = len(self.data)
        if l==0:
            return None
        i = randint(0, l-1)
        res = copy(self.data[i][1])
        del self.data[i]
        return res


def mpi_svpchallenge_dist_parrallel_asvp(A, bs_max, goal, cores):
    POOL_SIZE = 8 * cores
    POOL_COPIES = 1 + cores/4
    n = A.nrows
    trials = cores * [0]
    As = cores * [None]
    if (rank == 0):
        print "# [per-MPI] Pool_size = ", POOL_SIZE
        print "# [per-MPI] Pool_copies = ", POOL_COPIES    
        print "# [per-MPI] trials = ", trials
        print "# [per-MPI] As = ", As
        print "# [per-MPI] Goals =", goal

    # randomization in the first place
    for i in range(cores):
        As[i] = copy_to_IntegerMatrix_long (A)
        bkz = BKZReduction(As[i])
        bkz.randomize_block(0, n, density=n/4)
        del bkz

    sv_pool = SVPool(POOL_SIZE, copies=POOL_COPIES)
    workers = cores*[None]
    over = False

    print "### Process", rank, " starts cores ", cores
    filenames = ["b" + str(n) + "_n" + str(rank) + \
                     "-c" + str(i) for i in xrange(cores)]

    
    # list of queues
    list_queue = [Queue() for i in xrange(cores)]
    while not over:

        state = MPI.Status()
        okay = comm.iprobe(source=MPI.ANY_SOURCE, tag=0, status=state)
        if(okay):
            node = state.Get_source()
            data = comm.recv(source=node, tag=0)
            #print "#  received end signal ", data, "from node ", node, " terminate now ... "
            if (data == -1):
                return -1
        
        sleep(.01)
        for i in range(cores):

            if workers[i] is None:
                v = sv_pool.pop()
                if v is not None:
                    #print "POPPED"
                    insert_in_IntegerMatrix(As[i], v)
                pickle.dump((As[i], False), open(filenames[i], 'wb'))
                    
                bsi = bs_max - 20
                bsi += min(20, 2*trials[i])
                #BS_RANGE = 10
                bsi -= randint(0, 5)
                workers[i] = Process(target=svp_improve_trial, args=(As[i], bsi, list_queue[i], filenames[i]))
                t = workers[i].start()
                
            if (workers[i] is not None) and (not workers[i].is_alive()):

                As[i], success = pickle.load(open(filenames[i], 'rb'))
                #As[i] = list_queue[i].get()
                #success = list_queue[i].get()                
                if (success):
                    sv_pool.push([x for x in As[i][0]])
                workers[i] = None
                trials[i] += 1
                norm = sum([x*x for x in As[i][0]])
                #print "#  sv found ", norm
                if norm < goal:
                    print "# SVP-", n, "SOLUTION :", As[i][0]
                    print "# Norm = ", norm, " < goal = ", goal
                    over = True
                    break

    for w in [w for w in workers if w is not None]:
        w.terminate()

    while True:
        sleep(.01)
        some_alive = False
        for w in [w for w in workers if w is not None]:
            some_alive |= w.is_alive()
        if not some_alive:
            return 1


###########################################################################
def main():
    # parse argument
    args = parse_options()
    if rank == 0:
        print "###################################### "
        print ("# [Args] MPI: %d" % size)
        print ("# [Args] file: %s" % args.file)
        print ("# [Args] block_size: %s" % args.bs_diff)
        print ("# [Args] cores per MPI: %s" % args.cores)
        print "###################################### "
    print("### initialization of MPI from %s %d of %d" % (machine, rank, size))
        
        
    # start process
    bs_diff = int(args.bs_diff)
    cores = int(args.cores)
    filename = args.file

    # read matrix
    try:
        with open(filename, "rb") as f:
            mat = pickle.load(f)
        if isinstance(mat, IntegerMatrix):
            A_input = mat
        else:
            A_input = IntegerMatrix.from_matrix(mat)
    except:
        A_input = IntegerMatrix.from_file(filename)

    # start svp reduction
    dim = A_input.nrows
    if (rank == 0):
        print "# input dim: ", dim
        print "# nrows: ", A_input.nrows
    ASVP_START = time()
    LLL.reduction(A_input)
    A = IntegerMatrix.from_matrix(A_input, int_type="long")
    bkz = BKZReduction(A)
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(dim)]
    goal = (1.05)**2 * gaussian_heuristic(r)
    bs_ulim = dim - bs_diff
    re = mpi_svpchallenge_dist_parrallel_asvp(A, bs_ulim, goal, cores)
    ASVP_TIME = time() - ASVP_START

    if (re == 1):
        print ("\nSUMMARY", {"input dim": dim, "bs_range": (bs_ulim-bs_diff, bs_ulim), "time": ASVP_TIME})

    if (re == 1):        
        for i in range(size):
            if (i != rank):
                endflag = -1
                comm.isend(endflag, dest=i, tag=0)

    return
        
if __name__ == '__main__':
    main()
    
