# -*- coding: utf-8 -*-
"""
Parallel BKZ reduction.
# can't remember what I wanted to do here -- to be recalled ...
"""
import numpy as np
import random
from multiprocessing import Process, Queue
import os, sys, pickle, numpy, platform
from copy import copy, deepcopy
from time import sleep, clock, time
from math import log, sqrt, floor
from time import time
from time import clock
import ctypes, os

# fpylll
from fpylll import IntegerMatrix, LLL, GSO
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz import BKZReduction as BKZ1
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
#from fpylll.algorithms.bkz2_otf_subsol import BKZReduction as BKZ2_SUB
from bkz2_otf_subsol_mod import BKZReduction as BKZ2_SUB
from fpylll.tools.bkz_stats import BKZTreeTracer, dummy_tracer
from fpylll.util import adjust_radius_to_gh_bound
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning
from fpylll.util import gaussian_heuristic
from fpylll.tools.bkz_stats import BKZTreeTracer, dummy_tracer
# mpi
from mpi4py import MPI
###########################################
# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
machine = platform.node()

BOUND_SINGLE = 80
BS_SELECTIVE = 190
THRESHOLD_LEVEL1 = 20
THRESHOLD_LEVEL2 = THRESHOLD_LEVEL1 * 3

###########################################
GRADIENT_BLOCKSIZE = 31
SUBSOL_BLOCKSIZE = 41
NPS = 60*[2.**29] + 5 * [2.**27] + 5 * [2.**26] + 1000 * [2.**25]
TYPE='double'
BS_DIFF = 50


class SVPool:
    def __init__(self, max_len, copies=1):
        self.max_len = max_len
        self.copies = copies
        self.data = []

    def push(self, v, norm):
        for i in range(self.copies):
            self.data += [copy((norm, v))]
        if len(self.data)> self.max_len:
            self.data.sort()

    def pop(self):
        l = len(self.data)
        if l==0:
            return None
        i = random.randint(0, l-1)
        res = copy(self.data[i][1])
        del self.data[i]
        return res


class BKZReduction(BKZ2):
    """
    BKZ 2.0 with parallel SVP reduction.
    """
    def __init__(self, A, ncores=2):
        """Create new BKZ object.

        :param A: an integer matrix, a GSO object or an LLL object
        :param ncores: number of cores to use

        """
        self.ncores = ncores
        BKZ2.__init__(self, A)


    def compare (self, A, B):
        diff = []
        for i in range(A.nrows):
            if (list(A[i]) != list(B[i])):
                diff.append(i)
                #print " FOUND DIFF at i = ", i
                #print A[i]
                #print B[i]
        return diff

    def get_pruning(self, kappa, block_size, params, target, preproc_cost, tracer=dummy_tracer):

        radius = self.M.get_r(kappa, kappa) * self.lll_obj.delta
        r = [self.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
        gh_radius = gaussian_heuristic(r)

        #if (params.flags & BKZ.GH_BND and block_size > 30):
        radius = min(radius, gh_radius * params.gh_factor)
            
        preproc_cost += .001
        if not (block_size > GRADIENT_BLOCKSIZE):
            pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target, flags=0)
        else: 
            try:
                #pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target)
                #pruning = prune(radius, NPS[block_size] * preproc_cost, [r], 10,
                #                    metric="solutions", float_type="dd",
                #                    flags=Pruning.GRADIENT|Pruning.NELDER_MEAD)
                pruning = Pruning.run(radius, NPS[block_size] * preproc_cost, [r],
                                          0.1, flags=Pruning.NELDER_MEAD, float_type="double")
            except:
                pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target, flags=0)
        return radius, pruning
    
    
    def svp_preprocessing(self, kappa, block_size, params, trials, tracer=dummy_tracer):
        clean = True
        lll_start = kappa if ((params.flags & BKZ.BOUNDED_LLL) or trials>0) else 0
        with tracer.context("lll"):
            self.lll_obj(lll_start, lll_start, kappa + block_size)
            if self.lll_obj.nswaps > 0:
                clean = False
        if trials < 3:
            return clean
        
        shift = trials - 3 
        last_preproc = 2*(block_size/5) + shift + min(shift, 5)
        last_preproc = min(last_preproc, block_size - 10)
        preprocs = [last_preproc]
        for preproc in preprocs:
            prepar = params.__class__(block_size=preproc, flags=BKZ.BOUNDED_LLL)
            clean &= self.tour(prepar, kappa, kappa + block_size, tracer=tracer)

        return clean

  
    def insert_sub_solutions(self, kappa, block_size, sub_solutions):
        M = self.M
        l = len(sub_solutions)
        n = M.d
        assert l < block_size
        
        for (a, vector) in sub_solutions:
            M.create_row()
            if len(vector)==0:      # No subsolution at this index. Leaving a 0 vector

                with M.row_ops(M.d-1, M.d):
                    M.row_addmul(M.d-1, kappa, 1)
                continue 
            with M.row_ops(M.d-1, M.d):
                for i in range(block_size):                    
                    M.row_addmul(M.d-1, kappa + i, vector[i])    

        for k in reversed(range(l)):
            M.move_row(M.d-1, kappa + k)
            
        self.lll_obj(kappa, kappa, kappa + block_size + l)
        
        for i in range(l):
            M.move_row(kappa + block_size, M.d-1)
            M.remove_last_row()
            
        return


    def copy_to_IntegerMatrix_long(self, A):
        n = A.nrows
        l = A.ncols
        AA = IntegerMatrix(n, l, int_type="long")
        for i in xrange(n):
            for j in xrange(l):
                AA[i, j] = A[i, j]
        return AA

    
    def insert_in_IntegerMatrix(self, A, v, kappa, block_size):
        if (list(A[kappa]) == list(v)):
            return
        n = A.nrows
        l = A.ncols

        AA = IntegerMatrix(n + 1, l, int_type="long")
        for i in xrange(kappa):
            for j in xrange(l):
                AA[i, j] = A[i, j]
        for j in xrange(l):
            AA[kappa, j] = v[j]
        for i in xrange(kappa+1,n+1):
            for j in xrange(l):
                AA[i, j] = A[i-1, j]

        M = GSO.Mat(AA, float_type=TYPE)
        M.update_gso()
        bkz = BKZ2(M)
        try:
            bkz.lll_obj(kappa, kappa, kappa + block_size + 1) # longer: 1 more row
        except:
            pass

        index = 0
        for i in range(kappa, kappa + block_size + 1):
            if (AA[i].is_zero()):
                index = i - kappa
                break
            
        for i in xrange(kappa + index):
            for j in xrange(l):
                A[i, j] = AA[i, j]
        for i in xrange(kappa + index + 1, n+1):
            for j in xrange(l):
                A[i-1, j] = AA[i, j]
       
        """
        for i in range(n):
            bad = 0
            for j in range(l):
                if (A[i][j] == 0):
                    bad = bad + 1
                if (bad == n):
                    print " inserting ", v
                    print " <<<<<<<<<<<<<<<<<<<<<< WRONG HERE at ", kappa, block_size, i
                    print A
                    print "old A is "
                    print AA_old
                    print "old A after lll is "                    
                    print AA                    
                    sys.exit(1)
        """

        del bkz
        del AA
        del M        
        return


    ###########################################################################
    # no rereandomizaiton, use insertion subsolutions
    def svp_reduction_single(self, kappa, block_size, params, tracer=dummy_tracer):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """
        
        verbose = 0
        if (verbose):
            start_time = time()
            self.M.update_gso()
            r = [self.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
            gh_length = gaussian_heuristic(r) * params.gh_factor
            kappa_length = self.M.get_r(kappa, kappa) 
            goal = min(kappa_length, gh_length)

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)
        remaining_probability = 1.0
        rerandomize = False
        trials = 0
        sub_solutions = block_size > SUBSOL_BLOCKSIZE
        while remaining_probability > 1. - params.min_success_probability:

            # 1. preprocessing
            preproc_start = time()
            with tracer.context("preprocessing"):
                self.M.update_gso()        
                self.svp_preprocessing(kappa, block_size, params, trials, tracer=tracer)
            preproc_cost = time() - preproc_start

            with tracer.context("pruner"):
                target = 1 - ((1. - params.min_success_probability) / remaining_probability)

                radius, pruning = self.get_pruning(kappa, block_size, params, target*1.01, 
                                                    preproc_cost, tracer)

            # 2. enum
            
            enum_obj = Enumeration(self.M, sub_solutions=sub_solutions)
            try:
                with tracer.context("enumeration",
                                    enum_obj=enum_obj,
                                    probability=pruning.expectation,
                                    full=block_size==params.block_size):
                    max_dist, solution = enum_obj.enumerate (kappa, kappa + block_size,
                        radius, 0, pruning=pruning.coefficients)[0]

                # 3. post processing
                with tracer.context("postprocessing"):
                    preproc_start = time() # Include post_processing time as the part of the next pre_processing
                    
                    if not sub_solutions:
                        self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)
                    if sub_solutions:
                        self.insert_sub_solutions(kappa, block_size, enum_obj.sub_solutions[:1+block_size/4])
                    self.M.update_gso()

            except EnumerationError:
                preproc_start = time()
                
            remaining_probability *= (1 - pruning.expectation)
            trials += 1

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)
        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        if (verbose):        
            if (rank ==0):
                kappa_length = self.M.get_r(kappa, kappa)
                print "# [rank %d] kappa %d, bs %d, r %d (gh %d), time %s, trials %s " % \
                  (rank, kappa, block_size, kappa_length, goal, time()-start_time, trials)
        
        return clean

    
    ###########################################################################
    def svp_reduction_single_nosub(self, kappa, block_size, params, tracer=dummy_tracer):
        """
        :param kappa:
        :param block_size:
        :param params:
        :param tracer:
        """

        if (block_size == 80):
            start_time = clock()

        
        #if block_size < 30:
        #    return BKZBase.svp_reduction(self, kappa, block_size, params, tracer=tracer)
        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize, trials = 1.0, False, 0

        while remaining_probability > 1. - params.min_success_probability:
            preproc_start = clock()    
            with tracer.context("preprocessing"):
                if False: # ((trials%5)==4):
                    self.randomize_block(kappa+1, kappa+block_size, density=1, tracer=tracer)
                self.svp_preprocessing(kappa, block_size, params, tracer=tracer, trials=trials)
            preproc_cost = clock() - preproc_start

            with tracer.context("pruner"):
                target = 1 - ((1. - params.min_success_probability) / remaining_probability)
                target =  min(target, .5)
                # target = params.min_success_probability
                radius, pruning = self.get_pruning(kappa, block_size, params, target*1.01, preproc_cost, tracer)

            if (block_size == 90):
                print " single target is ", target, ", expectation ", pruning.expectation
                

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration",
                                    enum_obj=enum_obj,
                                    probability=pruning.expectation,
                                    full=block_size==params.block_size):
                    max_dist, solution = enum_obj.enumerate(kappa, kappa + block_size, radius, 0,
                                                            pruning=pruning.coefficients)[0]
                with tracer.context("postprocessing"):
                    rerandomize = True
                    self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)

            except EnumerationError:
                rerandomize = False

            remaining_probability *= (1 - pruning.expectation)
            trials += 1

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        if (block_size == 90):
            print " time bs 90 is ",    clock() - start_time, ", trials = ", trials, kappa
        
        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean



    ###########################################################################
    # used as backup
    def svp_reduction_mpi_other(self, As, kappa, block_size, params, queue):
        
        verbose = 0
        M = GSO.Mat(As, float_type=TYPE)
        params = fplll_bkz.Param(block_size=block_size, max_loops=2,
                                     min_success_probability=.01, flags=BKZ.BOUNDED_LLL)
        bkz_sub = BKZ2_SUB(M)
        bkz_sub.M.update_gso()

        if (verbose):
            ENUM_START = time()
        bkz_sub.lll_obj(kappa, kappa, kappa + block_size)
        r_old = bkz_sub.M.get_r(kappa, kappa)
        bkz_sub.svp_reduction(kappa, block_size, params, tracer=dummy_tracer)
        #bkz_sub.M.update_gso()
        r_new = bkz_sub.M.get_r(kappa, kappa)
        length = -1
        success = False
        if (r_new < r_old):
            success = True
            length = r_new
        if (verbose):
            ENUM_TIME = time() - ENUM_START
            print "# oth-SVP-%d, proc %s, time %.2f, norm %d" %(block_size, os.getpid(), ENUM_TIME, length)

        queue.put(success)
        queue.put(length)

        for i in range(block_size):
            while (queue.qsize() != 0):
                continue
            print " putting ", As[kappa+i]
            queue.put(As[kappa+i])

        
        return


    ###########################################
    def svp_reduction_mpi_trial_enum (self, bkz_sub, preproc_cost, radius, kappa, block_size):
        verbose = 0
        bkz_sub.M.update_gso()
        r = [bkz_sub.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
        r_old = r[0]
        gh = gaussian_heuristic(r)
        PRUNE_START = time()
        try:
            pruning = prune(radius, NPS[block_size] * preproc_cost, [r], 10,
                                metric="solutions", float_type="mpfr",
                                flags=Pruning.GRADIENT|Pruning.NELDER_MEAD)
            """
            pruning = prune(radius, NPS[block_size] * preproc_cost, [r], 0.0001,
                                metric="probability", float_type="mpfr",
                                flags=Pruning.GRADIENT|Pruning.NELDER_MEAD)            
            """
        except:
            return False, -1, 0, 0, 0
        PRUNE_TIME = time() - PRUNE_START
        ENUM_START = time()
        enum_obj = Enumeration(bkz_sub.M, sub_solutions=True)
        success = False
        length = -1
        #print radius, pruning.coefficients
        estimate_cost = sum(pruning.detailed_cost) / NPS[block_size]
        try:        
            enum_obj.enumerate(kappa, kappa+block_size, radius, 0, pruning=pruning.coefficients)
            length = enum_obj.sub_solutions[0][0]
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
        bkz_sub.M.update_gso()
        #A_old = deepcopy(bkz_sub.A)
        bkz_sub.insert_sub_solutions(kappa, block_size, enum_obj.sub_solutions[:1+block_size/4])
        #print self.compare(A_old, bkz_sub.A)
        bkz_sub.M.update_gso()
        r_new = bkz_sub.M.get_r(kappa, kappa)
        if (r_new < r_old):
            success = True
            length = r_new
        
        return success, length, PRUNE_TIME, ENUM_TIME, estimate_cost


    ###########################################
    def svp_reduction_mpi_trial_old(self, As, kappa, block_size, bs, queue, filename):
        verbose = 0

        # setup gso
        n = As.nrows
        M = GSO.Mat(As, float_type=TYPE)
        bkz_sub = BKZ2_SUB(M)
        bkz_sub.M.update_gso()

        # sanity check
        if (bs <= 4):
            #queue.put(As)
            #queue.put(False)
            #norm = bkz_sub.M.get_r(kappa, kappa)            
            #queue.put(norm)
            return

        # do lll
        bkz_sub.lll_obj(kappa, kappa, kappa + block_size)
        # preprocessing
        BKZ_START = time()
        bkz_sub.M.update_gso()
        for lbs in range(30, bs, 2) + [bs]:
            params = fplll_bkz.Param(block_size=lbs, max_loops=10,
                                         min_success_probability=.5,
                                         flags=BKZ.BOUNDED_LLL) # bounded LLL is important
            bkz_sub (params, kappa, kappa + block_size)
            bkz_sub.lll_obj (kappa, kappa, kappa + block_size)
            bkz_sub.M.update_gso()
        BKZ_TIME = time() - BKZ_START
        if (verbose):
            #print "#  BKZ-%d, proc %s, time %.2f" %(bs, os.getpid(), BKZ_TIME)  
            ENUM_START = time()
        
        r = bkz_sub.M.get_r(kappa, kappa)
        success, length, prune_time, enum_time, estimate_cost = self.svp_reduction_mpi_trial_enum (bkz_sub, BKZ_TIME, r*.99, kappa, block_size)
        if (verbose):
            ENUM_TIME = time() - ENUM_START
            print "#   [rank %d] BKZ-%d trial: Tbkz %.1f, Tsvp %.1f, (prune %.1f, enum %.1f, est. %.1f), norm %d" %(rank, bs, BKZ_TIME, ENUM_TIME, prune_time, enum_time, estimate_cost, length)

        bkz_sub.M.update_gso()
        norm = bkz_sub.M.get_r(kappa, kappa)

        """
        queue.put(success)
        queue.put(norm)
        mat = [[0 for _ in range(As.ncols)] for _ in range(block_size)]
        As[kappa:kappa+block_size].to_matrix(mat)
        for i in range(block_size):
            queue.put(mat[i])
        """
        pickle.dump((As, norm, success), open(filename, 'wb'))
        
        del bkz_sub
        del M
        
        return 
    
    
    ###########################################
    def svp_reduction_mpi_trial(self, As, kappa, block_size, bs, queue, filename):
        verbose = 0

        # setup gso
        n = As.nrows
        M = GSO.Mat(As, float_type=TYPE)
        bkz_sub = BKZ2_SUB(M)
        bkz_sub.M.update_gso()

        # sanity check
        if (bs <= 4):
            #queue.put(As)
            #queue.put(False)
            #norm = bkz_sub.M.get_r(kappa, kappa)            
            #queue.put(norm)
            return

        # do lll
        bkz_sub.lll_obj(kappa, kappa, kappa + block_size)
        bkz_sub.M.update_gso()
        r_initial = bkz_sub.M.get_r(kappa, kappa)
        
        # preprocessing
        success = False
        for lbs in range(30, bs, 2) + [bs]:
            params = fplll_bkz.Param(block_size=lbs, max_loops=10,
                                         min_success_probability=.5,
                                         flags=BKZ.BOUNDED_LLL) # bounded LLL is important
            BKZ_START = time()
            bkz_sub (params, kappa, kappa + block_size)
            bkz_sub.lll_obj (kappa, kappa, kappa + block_size)
            print "#   [rank %d, proc %d] BKZ-%d " %(rank, os.getpid(), lbs)
            bkz_sub.M.update_gso()
            r = bkz_sub.M.get_r(kappa, kappa)
            BKZ_TIME = time() - BKZ_START

            if (verbose):
                #print "#  BKZ-%d, proc %s, time %.2f" %(bs, os.getpid(), BKZ_TIME)  
                ENUM_START = time()
            
            success, length, prune_time, enum_time, estimate_cost = self.svp_reduction_mpi_trial_enum (bkz_sub, BKZ_TIME, r*.99, kappa, 148)

            bkz_sub.M.update_gso()
            r_new = bkz_sub.M.get_r(kappa, kappa)
            
            if (r_new < r_initial):
                break
            # rerandomize if failed
            bkz_sub.randomize_block(kappa, kappa+block_size, density=20)
            
        if (verbose):
            ENUM_TIME = time() - ENUM_START
            print "#   [rank %d] BKZ-%d trial: Tbkz %.1f, Tsvp %.1f, (prune %.1f, enum %.1f, est. %.1f), norm %d" %(rank, bs, BKZ_TIME, ENUM_TIME, prune_time, enum_time, estimate_cost, length)

        bkz_sub.M.update_gso()
        norm = bkz_sub.M.get_r(kappa, kappa)

        """
        queue.put(success)
        queue.put(norm)
        mat = [[0 for _ in range(As.ncols)] for _ in range(block_size)]
        As[kappa:kappa+block_size].to_matrix(mat)
        for i in range(block_size):
            queue.put(mat[i])
        """
        pickle.dump((As, norm, success), open(filename, 'wb'))
        
        del bkz_sub
        del M
        
        return 
    

    ###########################################################################
    def svp_reduction_mpi(self, kappa, block_size, params, tracer=dummy_tracer):
        # time
        start_time = time()

        # max preprocessing block size
        bs_diff = BS_DIFF
        bs_max = block_size - bs_diff

        # set goal
        self.M.update_gso()

        # current vector
        #kappa_length = self.M.get_r(kappa, kappa) * self.lll_obj.delta
        kappa_length = self.M.get_r(kappa, kappa)
        
        # gh length
        r = [self.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
        gh_length = gaussian_heuristic(r) * params.gh_factor
        
        goal = gh_length 
        if (kappa_length <= goal):
            if (rank ==0 and block_size >= BOUND_SINGLE): 
                print "# [rank %d] kappa = %d bs = %d, goal = %d, r = %d (already achieved -- pass)" % \
                  (rank, kappa, block_size, goal, r[0])
            return 1

        """
        goal = (goal + kappa_length) / 2
        if (goal < r[0] * 0.95):
            goal = r[0] * 0.95
        """
        
        # info
        #"""
        if (rank ==0 and block_size >= BOUND_SINGLE):        
            print "# [rank %d] kappa = %d bs = %d, goal = %d, r = %d" % \
              (rank, kappa, block_size, goal, r[0])

              
        #"""
        # set matrices
        n = self.A.nrows
        trials = self.ncores * [0]
        As = self.ncores * [None]
        POOL_SIZE = 8 * self.ncores
        POOL_COPIES = 1 + self.ncores/4

        # randomization matrices
        for i in range(self.ncores):
            As[i] = self.copy_to_IntegerMatrix_long (self.A)
            M = GSO.Mat(As[i], float_type=TYPE)
            bkz = BKZ2_SUB(M)
            bkz.randomize_block(kappa, kappa+block_size, density=block_size/3)
            del bkz
            del M

        # setup share pools
        sv_pool = SVPool(POOL_SIZE, copies=POOL_COPIES)
        sv_pool.data = []
        workers = self.ncores*[None]
        over = False
        
        #print "####################################################"
        #print "### Process", rank, " starts cores ", self.ncores

        # list of queues
        list_queue = [Queue() for i in xrange(self.ncores)]
        fail_list = [0 for i in xrange(self.ncores)]
        break_flag = [0 for i in xrange(self.ncores)]
        terminated_by_other = 0
        filenames = ["b" + str(n) + "_n" + str(rank) + \
                    "-c" + str(i) for i in xrange(self.ncores)]
                    
        while not over:
            
            state = MPI.Status()
            okay = comm.iprobe(source=MPI.ANY_SOURCE, tag=99, status=state)
            if(okay):
                node = state.Get_source()
                data = comm.recv(source=node, tag=99)
                #print "# [rank %d] receive exit signal from rank %d " % (rank, node)
                if (data == 1):
                    terminated_by_other = 1
                    self.A = As[0] # need improve
                    self.M = GSO.Mat(self.A, float_type=TYPE)
                    self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT) 
                    for i in range(self.ncores):
                        self.insert_in_IntegerMatrix(self.A, As[i][kappa], kappa, block_size)
                    break
           
            sleep(0.01)
            
            for i in range(self.ncores):
                
                if workers[i] is None:
                    v = sv_pool.pop()
                    if v is not None:
                        self.insert_in_IntegerMatrix(As[i], v, kappa, block_size)

                    bsi = bs_max - 30
                    bsi += min(20, 2*trials[i]) # about 10 trials to go up to bs_max
                    bsi -= random.randint(0, 3)

                    norm = self.M.get_r(kappa, kappa)
                    pickle.dump((As[i], norm, False), open(filenames[i], 'wb'))

                    # just use current bsi
                    if (fail_list[i] <= 0):
                        workers[i] = Process(target=self.svp_reduction_mpi_trial,
                                                 args=(As[i], kappa, block_size,
                                                           bsi, list_queue[i],
                                                           filenames[i]))

                    # increase blocksize bsi                      
                    elif (fail_list[i] > 0 and fail_list[i] <= THRESHOLD_LEVEL1):
                        bsi = min(bs_max, bsi + fail_list[i])
                        workers[i] = Process(target=self.svp_reduction_mpi_trial,
                                                 args=(As[i], kappa, block_size,
                                                           bsi, list_queue[i],
                                                           filenames[i]))
                    # stablize bsi; but rerandomization                        
                    elif (fail_list[i] <= THRESHOLD_LEVEL2):
                        M = GSO.Mat(As[i], float_type=TYPE)
                        bkz = BKZ2_SUB(M)
                        bkz.randomize_block(kappa, kappa+block_size, density=block_size/3)
                        del bkz
                        del M
                        workers[i] = Process(target=self.svp_reduction_mpi_trial,
                                                 args=(As[i], kappa, block_size,
                                                           bsi, list_queue[i],
                                                           filenames[i]))
                    else:
                        workers[i] = Process(target=self.svp_reduction_mpi_trial,
                                                 args=(As[i], kappa, block_size,
                                                           bsi, list_queue[i],
                                                           filenames[i]))
                        break_flag[i] = True
                    
                    # start woker
                    t = workers[i].start()

                if (workers[i] is not None) and (not workers[i].is_alive()):

                    # get and insert
                    As[i], norm, success = pickle.load(open(filenames[i], 'rb'))
                    
                    """
                    success = list_queue[i].get()
                    norm = list_queue[i].get()
                    j = 0
                    # buffer read for queue
                    while True:
                        if (j >= block_size):
                            break
                        try:
                            v = list_queue[i].get()
                        except Queue.Empty:
                            break
                        for k in range(As[i].ncols):
                            As[i][kappa+j, k] = v[k]
                        j += 1
                    #for j in xrange(THRESHOLD_SEND):
                    #    self.insert_in_IntegerMatrix(As[i], mat[j], kappa, block_size)
                    """

                    # break
                    if (break_flag[i]):
                        workers[i] = None
                        self.A = As[i]
                        self.M = GSO.Mat(self.A, float_type=TYPE)
                        self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
                        over = True
                        break
                    
                    # found smaller norm than goal
                    if norm < goal:
                        #print "# SVP-", n, "SOLUTION :", As[i][kappa]
                        #print "#   [rank %d] found SVP-%d, norm %d < goal %d" % (rank,
                        #            block_size, norm, goal)
                        self.A = As[i]
                        self.M = GSO.Mat(self.A, float_type=TYPE)
                        self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
                        over = True
                        break
                    # found smaller norm but larger than goal
                    if (success):
                        sv_pool.push([x for x in As[i][kappa]], norm)
                        fail_list[i] = 0
                    else:
                        norm = 1e100
                        fail_list[i] += 1

                    # done update
                    workers[i] = None
                    trials[i] += 1
                    
        for w in [w for w in workers if w is not None]:
            w.terminate()
            
        # terminated by this process -- signal others MPI's to stop
        if (terminated_by_other==0):
            for i in range(size):
                if (i != rank):
                    comm.isend(1, i, tag=99)

        # sending data to master node
        self.M.update_gso()
        if (rank != 0):
            send_vec = []
            #for i in range(block_size/10):
            #    send_vec.append(list(self.A[kappa+i]))
            send_vec.append(list(self.A[kappa]))
            # self.M.update_gso()
            norm = self.M.get_r(kappa, kappa)
            comm.isend(send_vec, dest=0, tag=11)

            
        # master node receiving data
        if (rank == 0):
            num = 0
            while(1):
                if (num >= size-1):
                    break
                sleep(.01)
                state = MPI.Status()
                okay = comm.iprobe(source=MPI.ANY_SOURCE, tag=11, status=state)
                if(okay):
                    num += 1 
                    node = state.Get_source()
                    vectors = comm.recv(source=node, tag=11)
                    for i in range(len(vectors)):
                        self.insert_in_IntegerMatrix(self.A, vectors[i], kappa, block_size)
                    self.M = GSO.Mat(self.A, float_type=TYPE)
                    self.M.update_gso()
                    self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
                    norm = self.M.get_r(kappa, kappa)
                    
                    
        # message
        self.M.update_gso()
        kappa_length = self.M.get_r(kappa, kappa)
        if (rank ==0 and block_size >= BOUND_SINGLE):
            print "# [rank %d] kappa %d, bs %d, r %d (gh %d), time %s, trials %s " % \
            (rank, kappa, block_size, kappa_length, goal, time()-start_time, trials)
            
        # check processes
        while True:
            sleep(.01)
            some_alive = False
            for w in [w for w in workers if w is not None]:
                some_alive |= w.is_alive()
            if not some_alive:
                return 1

    ###########################################################################            
    def svp_reduction_call_single (self, kappa, block_size, params, tracer=dummy_tracer):
        
        clean = True
        if (rank == 0):
            # in such case, signal slaves to skip
            for i in range(1,size):
                comm.send(-1, dest=i, tag=0)
            
            clean = self.svp_reduction_single(kappa, block_size, params, tracer=tracer)
            
        return clean

    
    ###########################################################################        
    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        
        """
        SVP reduction attempts until the probability threshold is reached.

        :param kappa: current index
        :param block_size: block size
        :param params: BKZ parameters
        :param tracer: object for maintaining statistics

        .. note: This function uses for to parallelise.

        """
        
        # correct block_size if needed
        block_size = min(block_size, self.A.nrows - kappa)

        # size reduction
        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        # if too small, used single svp (this has to be in front of the following)
        if block_size <= BOUND_SINGLE:
            return self.svp_reduction_call_single(kappa, block_size, params, tracer=tracer)
        
        # sending start signal and params
        for i in range(1,size):
            comm.send(1, dest=i, tag=0)
        for i in range(1,size):
            comm.send(self.A, dest=i, tag=1)
        for i in range(1,size):
            comm.send(kappa, dest=i, tag=2)

        # start svp
        self.svp_reduction_mpi(kappa, block_size, params, tracer=tracer)

        if (rank == 0):
            mat = [[0 for _ in range(self.A.ncols)] for _ in range(self.A.nrows)]
            self.A.to_matrix(mat)
            filename = "%d.tmp"%(os.getpid())
            with open(filename, "wb") as f:
                 pickle.dump(mat, f)
            del mat

        # done here
        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean


    def select_index(self, block_size, min_row, max_row):
        self.M.update_gso()        
        r = self.M.r()
        maxv = -1
        for k in range(min_row, max_row-block_size-1):
            # current vector
            k_length = self.M.get_r(k, k)
            gh = gaussian_heuristic(r[k:k+block_size]) * 1.1
            v = sqrt(k_length) / sqrt(gh)
            if v > maxv:
                maxv, maxi = v, k
                
        if (maxv < 1):
            maxi = 0
                
        return maxi

    
    def selective_tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        if max_row == -1:
            max_row = self.A.nrows
        for i in range(min_row, max_row-1):
            self.M.update_gso()
            self.lll_obj.size_reduction(0, min_row+1)
            kappa = self.select_index(params.block_size,  min_row, max_row)
            print "# selecting index, ", kappa
            block_size = min(params.block_size, max_row - kappa)
            self.svp_reduction(kappa, block_size, params, tracer)
            self.lll_obj.size_reduction(0, min_row+1)            
        return False
        

    def usual_tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        """One BKZ loop over all indices.

        :param params: BKZ parameters
        :param min_row: start index ≥ 0
        :param max_row: last index ≤ n

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if max_row == -1:
            max_row = self.A.nrows

        clean = True
        for kappa in range(min_row, max_row-1):
            block_size = min(params.block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, block_size, params, tracer)

        self.lll_obj.size_reduction(max(0, max_row-1), max_row, max(0, max_row-2))

        r = [log(self.M.get_r(i, i)) for i in range(0, self.A.nrows)]

        if (rank == 0): 
            if (params.flags & BKZ.DUMP_GSO):
                print "# end of loop"
                print r
        
        return clean
    

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        if (params.block_size >= BS_SELECTIVE):
            return self.selective_tour(params, min_row, max_row, tracer)
        else:
            return self.usual_tour(params, min_row, max_row, tracer)









