# others
from copy import copy
from multiprocessing import Process
from time import sleep, clock, time
from random import randint
from math import log, sqrt, floor
import pickle
import os
import sys

# fpylll
from fpylll import IntegerMatrix, LLL, GSO
from fpylll import BKZ as fplll_bkz
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.algorithms.bkz2_otf_subsol import BKZReduction
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning
from fpylll.util import gaussian_heuristic


# params
"""
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
print ("# [Args] Cmd: %s" % str(sys.argv[0]))
print ("# [Args] Args: %s " % cmdargs)
print ("# [Args] Args[1]: %s" % str(sys.argv[1]))
print ("# [Args] Args[2]: %s" % str(sys.argv[2]))
print ("# [Args] Args[3]: %s" % str(sys.argv[3]))
"""
# python -u svpchallenge_par3.py bs_diff=10 cores=2 start_dim=80


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


def enum_trial(bkz, preproc_cost, radius):
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
    print ("# [Prune] time %.4f"%PRUNE_TIME)

    ENUM_TIME = time() - ENUM_START
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
    A, _ = pickle.load(open(filename, 'rb'))
    n = A.nrows
    bkz = BKZReduction(A)

    BKZ_START = time()
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(n)]
    print "*********************************************************"
    print "# Run with BS = %d" % bs
    print "# [ File", filename, "]", "before BKZ",
    print_basis_stats(bkz.M, n)
    gh = gaussian_heuristic(r)

    llbs = []
    for lbs in range(30, bs - 10, 2) + [bs]:
        llbs.append(lbs)
        params = fplll_bkz.Param(block_size=lbs, max_loops=1,
                                 min_success_probability=.01)
        bkz(params=params)
        bkz.lll_obj()
    print "# progressive block_sizes = ", llbs

    r = [bkz.M.get_r(i, i) for i in range(n)]

    BKZ_TIME = time() - BKZ_START
    print "# [ File", filename, "]", "after BKZ",
    print ("BKZ-[%d .. %d]  ... \t\t "%(30, bs)),
    print_basis_stats(bkz.M, n)
    print ("# [BKZ] TIME = %.2f"%BKZ_TIME)
    success = enum_trial(bkz, BKZ_TIME, r[0]*.99)
    print
    pickle.dump((A, success), open(filename, 'wb'))
    return success


class SVPool:
    def __init__(self, max_len, copies=1):
        self.max_len = max_len
        self.copies = copies
        self.data = []

    def push(self, v):
        #print "*** pushing pool size ", len(self.data)        
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
    

def interacting_parrallel_asvp(A, bs_ulim, goal, cores, BS_RANDOM_RANGE):
    POOL_SIZE = 8 * cores
    POOL_COPIES = 1 + cores/4
    n = A.nrows
    trials = cores * [0]
    As = cores * [None]
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

    while not over:
        sleep(.01)
        for i in range(cores):
            if workers[i] is None:
                v = sv_pool.pop()
                if v is not None:
                    insert_in_IntegerMatrix(As[i], v)
                bsi = bs_ulim - 20
                bsi += min(20, 2*trials[i])
                #bsi -= 2*randint(0, BS_RANDOM_RANGE/2)
                #print " pid is ",  os.getpid()
                pickle.dump((As[i], False), open("%d.%d.tmp"%(os.getpid(),i), 'wb'))
                workers[i] = Process(target=svp_improve_trial, args=("%d.%d.tmp"%(os.getpid(),i), bsi))
                workers[i].start()

            if (workers[i] is not None) and (not workers[i].is_alive()):
                As[i], success = pickle.load(open("%d.%d.tmp"%(os.getpid(),i), 'rb'))
                if success:
                    sv_pool.push([x for x in As[i][0]])
                workers[i] = None
                trials[i] += 1
                norm = sum([x*x for x in As[i][0]])
                if norm < goal:
                    print ("SVP-%d SOLUTION :"%n, As[i][0])
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

        
# svpchallenge_par3(bs_diff=10, cores=2, start_dim=80, end_dim=80+2, BS_RANDOM_RANGE = 10)
# svpchallenge_par3(bs_diff=10, cores=2, start_dim=40, end_dim=40+2, BS_RANDOM_RANGE = 10)
def svpchallenge_par3(bs_diff=10, cores=2, start_dim=80, end_dim=80+2, BS_RANDOM_RANGE = 10):
    for dim in range(start_dim, start_dim+2, 2):
        A_pre = IntegerMatrix.from_file("svpchallenge/svpchallengedim%dseed0.txt"%dim)
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
        interacting_parrallel_asvp(A, bs_ulim, goal, cores, BS_RANDOM_RANGE)
        ASVP_TIME = time() - ASVP_START

        print ("\nSUMMARY", {"input dim": dim, "bs_range": (bs_ulim - BS_RANDOM_RANGE, bs_ulim), "time": ASVP_TIME})




def svpchallenge_test ():
    dim = 60
    A_pre = IntegerMatrix.from_file("svpchallenge/svpchallengedim%dseed0.txt"%dim)
    print "# input dim: ", dim
    print "# nrows: ", A_pre.nrows
    ASVP_START = time()
    LLL.reduction(A_pre)
    A = IntegerMatrix.from_matrix(A_pre, int_type="long")
    bkz = BKZReduction(A)
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(dim)]
    goal = (1.05)**2 * gaussian_heuristic(r)
    params = fplll_bkz.Param(block_size=20, max_loops=1,
                                 min_success_probability=.01)
    bkz(params=params)
    print " done BKZ yes"
