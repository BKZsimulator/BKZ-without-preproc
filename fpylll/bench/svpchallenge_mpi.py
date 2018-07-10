# mpiexec -n 4 python svpchallenge_mpi.py bs_diff=10 cores=2 start_dim=80

# system
import os, sys, pickle, numpy, platform
from copy import copy
from multiprocessing import Process
from time import sleep, clock, time
from random import randint
from math import log, sqrt, floor
# fpylll
from fpylll import IntegerMatrix, LLL, GSO
from fpylll import BKZ as fplll_bkz
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.algorithms.bkz2_otf_subsol import BKZReduction
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning
from fpylll.util import gaussian_heuristic
# mpi
from mpi4py import MPI

###########################################
# params
total = len(sys.argv)
cmdargs = str(sys.argv)
bs_diff = str(sys.argv[1]).split("=")[1]
assert(str(sys.argv[1]).split("=")[0] == "bs_diff")
bs_diff = int(bs_diff)
cores = str(sys.argv[2]).split("=")[1]
assert(str(sys.argv[2]).split("=")[0] == "cores")
cores = int(cores)
start_dim = str(sys.argv[3]).split("=")[1]
assert(str(sys.argv[3]).split("=")[0] == "start_dim")
start_dim = int(start_dim)

###########################################
# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
machine = platform.node()


###########################################
if (rank == 0):
    print "###################################### "
    print ("# [Args] Cmd: %s" % str(sys.argv[0]))
    print ("# [Args] Args: %s " % cmdargs)
    print ("# [Args] Args[1]: %s" % str(sys.argv[1]))
    print ("# [Args] Args[2]: %s" % str(sys.argv[2]))
    print ("# [Args] Args[3]: %s" % str(sys.argv[3]))
    print "###################################### "
#print("### initialization of MPI from %s %d of %d" % (machine, rank, size))


###########################################
def copy_to_IntegerMatrix_long(A):
    n = A.nrows
    AA = IntegerMatrix(n, n, int_type="long")
    for i in xrange(n):
        for j in xrange(n):
            AA[i, j] = A[i, j]
    return AA


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


def print_basis_stats(M, n):
    r = [M.get_r(i, i) for i in range(n)]
    gh = gaussian_heuristic(r)
    lhvr = log(gaussian_heuristic(r[:n/2])) - log(gaussian_heuristic(r[n/2:]))
    #print (lhrv = %.4f, r[i]/gh)%lhvr,
    for i in range(20):
        print ('{}'.format(r[i]/gh)),
    print
    return


def insert_sub_solutions(bkz, sub_solutions):
    M = bkz.M
    l = len(sub_solutions)
    n = M.d
    count = 0
    for (a, vector) in sub_solutions:
        if len(vector)==0:      # No subsolution at this index. Leaving a 0 vector
            continue
        M.create_row()
        count = count + 1
        with M.row_ops(M.d-1, M.d):
            for i in range(n): 
                M.row_addmul(M.d-1, i, vector[i])
    for k in reversed(range(count)):
        M.move_row(M.d-1, k)
    bkz.lll_obj()
    for i in range(count):
        M.remove_last_row()
    return


def svp_improve_trial_enum(bkz, preproc_cost, radius):
    verbose = 0
    n = bkz.A.nrows
    r = [bkz.M.get_r(i, i) for i in range(0, n)]       
    gh = gaussian_heuristic(r)

    PRUNE_START = time()
    NPS = 2**24
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
    print ("# subsolutions : r[i]/gh"),
    for (a, b) in enum_obj.sub_solutions:
        print ("%.3f"%abs(a/gh)),
    print 
    """
    insert_sub_solutions(bkz, enum_obj.sub_solutions)    
    return success


def svp_improve_trial(filename, bs):

    verbose = 0
    A, _ = pickle.load(open(filename, 'rb'))
    n = A.nrows
    
    # 1. LLL
    bkz = BKZReduction(A)
    bkz.lll_obj()
    if (verbose):
        print "*********************************************************"
        print "# Run with BS = %d" % bs
        print "# [ File", filename, "]", "before BKZ",
        r = [bkz.M.get_r(i, i) for i in range(n)]
        print_basis_stats(bkz.M, n)

    # 2. progressive BKZ
    BKZ_START = time()
    llbs = []
    for lbs in range(30, bs - 10, 2) + [bs]:
        llbs.append(lbs)
        params = fplll_bkz.Param(block_size=lbs, max_loops=1,
                                 min_success_probability=.01)
        bkz(params=params)
        bkz.lll_obj()
    BKZ_TIME = time() - BKZ_START
    
    if (verbose):
        print "# progressive block_sizes = ", llbs
        print "# [ File", filename, "]", "after BKZ",
        print ("BKZ-[%d .. %d]  ... \t\t "%(30, bs)),
        print ("# [BKZ] TIME = %.2f"%BKZ_TIME)
        print_basis_stats(bkz.M, n)
        print

    # 3. enum
    r = [bkz.M.get_r(i, i) for i in range(n)]    
    success = svp_improve_trial_enum(bkz, BKZ_TIME, r[0]*.99)

    # 4. write
    pickle.dump((A, success), open(filename, 'wb'))
    print "# [cpu", rank, "] done BKZ with BS = %d" % bs    
    
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

    def first(self):
        l = len(self.data)
        if l==0:
            return None
        return copy(self.data[0])
    

def mpi_interacting_parrallel_asvp_push (A, sv_pool, break_flag):

    lst_comm = []

    # signal pushing to master
    comm.send(0, dest=0, tag=0)
    comm.send(rank, dest=0, tag=1)
    
    # always push the first one
    v = sv_pool.first()
    minnorm = sys.float_info.max
    if (v is not None):
        minnorm = v[0]
    sv_pool.push([x for x in A[0]])
    lst_comm.append(A[0])

    # and possibly others
    for i in range(A.nrows):
        minnorm = sv_pool.first()[0]
        norm = A[i].norm()**2
        if (norm < minnorm):
            sv_pool.push([x for x in A[i]])
            lst_comm.append(A[i])
            
    # send to head and then return
    if (lst_comm is not None):
        flag = 0
        n = len(lst_comm[0])
        d = len(lst_comm)
        data = numpy.arange(n*d, dtype='i')
        comm.send(d, dest=0, tag=2)
        for i in range(d):        
            for j in range(n):
                data[i*n+j] = lst_comm[i][j]
        comm.Send([data, MPI.INT], dest=0, tag=3)
    
        
def mpi_interacting_parrallel_asvp_pop (A, sv_pool, break_flag):

    # signal fetching from master
    comm.send(1, dest=0, tag=0)
    comm.send(rank, dest=0, tag=1)
    yes_receive = comm.recv(source=0, tag=8)

    # fetch data from master
    if (yes_receive == 1):
        d = comm.recv(source=0, tag=2)
        n = len(A[0])
        lst = numpy.empty(n*d, dtype='i')
        comm.Recv([lst, MPI.INT], source=0, tag=3)
        for i in range(0, len(lst), n):
            sv_pool.push([x for x in lst[i:i+n]])

    # start actual pop
    v = sv_pool.pop()
    if v is not None:
        insert_in_IntegerMatrix(A, v)

            
def mpi_interacting_parrallel_asvp(A, bs_ulim, goal, cores, BS_RANDOM_RANGE):
    POOL_SIZE = 8 * cores
    POOL_COPIES = 1 + cores/4
    n = A.nrows
    trials = cores * [0]
    As = cores * [None]
    if (rank == 0):
        print "# Pool_size = ", POOL_SIZE
        print "# Pool_copies = ", POOL_COPIES    
        print "# trials = ", trials
        print "# As = ", As
        print "# Goals =", goal
        
    for i in range(cores):
        As[i] = copy_to_IntegerMatrix_long(A)
        bkz = BKZReduction(As[i])
        bkz.randomize_block(0, n, density=n/4)
        # returns As[i], a randomized matrix
        del bkz
        
    sv_pool = SVPool(POOL_SIZE, copies=POOL_COPIES)
    workers = cores*[None]
    over = False

    print "### Process", rank, " starts slave process"

    if (rank == 0):
        rank0_time = time()
        print " time", rank0_time

    break_flag = 0
    while not over:
        if (rank == 0):
            rank0_curr_time = time()
            if ( rank0_curr_time - rank0_time > 1):
                rank0_time = rank0_curr_time
                print "  updating time", rank0_time
                v = sv_pool.first()
                if v is not None:
                    print "--------> first v is", v
                #data = comm.bcast(sv_pool, root=rank)
                #print "# rank ", rank, " collected ", data
        
        sleep(.01)
        for i in range(cores):
            
            # check return condition?
            if (break_flag):
                break
            
            if workers[i] is None:
                mpi_interacting_parrallel_asvp_pop (As[i], sv_pool, break_flag)
                bsi = bs_ulim - 20
                bsi += min(20, 2*trials[i])
                #bsi -= 2*randint(0, BS_RANDOM_RANGE/2)
                # write before staring
                pickle.dump((As[i], False), open("%d.%d.tmp"%(os.getpid(),i), 'wb'))
                workers[i] = Process(target=svp_improve_trial, args=("%d.%d.tmp"%(os.getpid(),i), bsi))
                workers[i].start()

            if (workers[i] is not None) and (not workers[i].is_alive()):
                As[i], success = pickle.load(open("%d.%d.tmp"%(os.getpid(),i), 'rb'))
                if success:
                    mpi_interacting_parrallel_asvp_push (As[i], sv_pool, break_flag)

                    
                workers[i] = None
                trials[i] += 1
                norm = sum([x*x for x in As[i][0]])
                if norm < goal:
                    print " # SVP-", n, "SOLUTION :", As[i][0]
                    print " # Norm = ", norm, " < goal = ", goal
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
            return


def mpi_svpchallenge_par3 (bs_diff=10, cores=2, start_dim=80, end_dim=80+2, BS_RANDOM_RANGE = 10):
    dim = start_dim
    A_pre = IntegerMatrix.from_file("/home/shi/suite/sb_fpylll/bench/svpchallenge/svpchallengedim%dseed0.txt"%dim)
    if (rank == 0):
        print "# input dim: ", dim
        print "# nrows: ", A_pre.nrows
    ASVP_START = time()
    LLL.reduction(A_pre)
    A = IntegerMatrix.from_matrix(A_pre, int_type="long")
    bkz = BKZReduction(A)
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(dim)]
    goal = (1.05)**2 * gaussian_heuristic(r)
    bs_ulim = dim - bs_diff
    mpi_interacting_parrallel_asvp(A, bs_ulim, goal, cores, BS_RANDOM_RANGE)
    ASVP_TIME = time() - ASVP_START
    
    # done send signal
    comm.send(99, dest=0, tag=0)
    comm.send(rank, dest=0, tag=1)

    if (rank == 0):
        print ("\nSUMMARY", {"input dim": dim, "bs_range": (bs_ulim - BS_RANDOM_RANGE, bs_ulim), "time": ASVP_TIME})

    return

def mpi_svpchallenge_par3_master (n):
    POOL_SIZE = 8 * cores
    POOL_COPIES = 1
    sv_pool_master = SVPool(POOL_SIZE, copies=POOL_COPIES)
    count = 0
    while (1):

        signal = comm.recv(source=MPI.ANY_SOURCE, tag=0)
        rank_rec = comm.recv(source=MPI.ANY_SOURCE, tag=1)
        
        # pushing
        if (signal == 0):
            d = comm.recv(source=rank_rec, tag=2)
            lst_comm = numpy.empty(n*d, dtype='i')
            comm.Recv([lst_comm, MPI.INT], source=rank_rec, tag=3)
            for i in range(0, len(lst_comm), n):
                sv_pool_master.push([x for x in lst_comm[i:i+n]])

        # popping
        elif (signal == 1):
            if (len(sv_pool_master.data) == 0):
                comm.send(0, dest=rank_rec, tag=8)                 
                continue
            comm.send(1, dest=rank_rec, tag=8) 
            lst = sv_pool_master.data
            n = len(lst[0][1])
            d = len(lst)
            data = numpy.arange(n*d, dtype='i')
            for i in range(d):
                for j in range(n):
                    data[i*n+j] = lst[i][1][j]
            comm.send(d, dest=rank_rec, tag=2)
            comm.Send([data, MPI.INT], dest=rank_rec, tag=3)
                
        elif (signal == 99):
            # one cpu success, return all
            comm.Abort()
            break
             

###########################################################################
if rank == 0:
    #mpi_svpchallenge_par3 (bs_diff, cores, start_dim, start_dim, 10)
    #comm.Send(randNum, dest=0)
    mpi_svpchallenge_par3_master (start_dim)
else:
    mpi_svpchallenge_par3 (bs_diff, cores, start_dim, start_dim, 10)    
        
"""
if rank == 0:
        print "Process", rank, "before receiving has the number", randNum[0]
        comm.Recv(randNum, source=1)
        print "Process", rank, "received the number", randNum[0]
"""
