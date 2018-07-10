from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
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
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.util import gaussian_heuristic
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning

preproc_cost = 2**40
NPS = 60*[2.**29] + 5 * [2.**27] + 5 * [2.**26] + 1000 * [2.**25]
block_size = 60

A = IntegerMatrix.random(60, "intrel", bits=600)
_ = LLL.reduction(A)
M = GSO.Mat(A)
_ = M.update_gso()
pr = Pruning.Pruner(M.get_r(0,0)*0.99, NPS[block_size]*preproc_cost, [M.r()], 0.0099, flags=0)
c = pr.optimize_coefficients([1. for _ in range(M.d)])
cost, details = pr.single_enum_cost(c, True)
print cost, details

kappa = 0
print "start"
print "kappa: ", kappa
NPS = 60*[2.**29] + 5 * [2.**27] + 5 * [2.**26] + 1000 * [2.**25]

params = BKZ.Param (block_size, max_loops=5000,
                        min_success_probability=.01,
                        flags=BKZ.BOUNDED_LLL, #|BKZ.DUMP_GSO,
                        dump_gso_filename="gso_output.file",
                        strategies = "default.json")

remaining_probability = 1.0
strategy = params.strategies[block_size]
radius = M.get_r(kappa, kappa) * 0.99
r = [M.get_r(i, i) for i in range(kappa, kappa+block_size)]
gh_radius = gaussian_heuristic(r)
radius = min(radius, gh_radius * params.gh_factor)
#pruning = strategy.get_pruning(radius, gh_radius)
preproc_cost = 2**20

total_cost_all = 0.0

while remaining_probability > (1. - params.min_success_probability):
#while True:
    target = 1 - ((1. - params.min_success_probability) / remaining_probability)
    print target
    pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target, flags=0)
    #print radius, pruning.coefficients
    print pruning.detailed_cost
#    estimate_cost = sum(pruning.detailed_cost) / NPS[block_size]
    estimate_cost = sum(pruning.detailed_cost)
    print estimate_cost
    probability = pruning.expectation

    total_cost = (estimate_cost+preproc_cost)/(probability)
    print preproc_cost, estimate_cost, total_cost, probability
    total_cost_all += total_cost
    remaining_probability *= (1 - pruning.expectation)
    print remaining_probability, params.min_success_probability
print total_cost_all
