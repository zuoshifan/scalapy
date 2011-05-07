
import time

import numpy as np

import scipy.linalg as la

import scarray

import os

matfile = "mat.dat"
evecsfile = "evecs.dat"
evalsfile = "evals.dat"

Br = 1000
Bc = 1000

Pr = 3
Pc = 3


#m1 = np.identity(2000, dtype=np.float64)
print "Generating matrix"

st = time.clock()
m1 = np.random.standard_normal((10000,10000))
m1 = 0.5*(m1 + m1.T)

m2 = scarray.matrix_pagealign(m1, [Br, Bc])

m2.reshape(-1, order='A').tofile(matfile)

et = time.clock()
print "Done. Time: %f" % (et-st)

print "Running ScaLapack"

st = time.clock()
os.system("mpirun -np %i ./evtest %i %i %i %i %i %i %s %s %s" % (Pr*Pc, m1.shape[0], m1.shape[1], Br, Bc, Pr, Pc, matfile, evalsfile, evecsfile))

et = time.clock()
print "Done. Time: %f" % (et-st)


print "Reading..."

st = time.clock()
evecst = np.fromfile(evecsfile, dtype=np.float64)

evecs1 = scarray.matrix_from_pagealign(evecst, m1.shape, [Br, Bc])

evals1 = np.fromfile(evalsfile, dtype=np.float64)

et = time.clock()
print "Done. Time: %f" % (et-st)


st = time.clock()
print "Numpy EV"

evals2, evecs2 = la.eigh(m1)


et = time.clock()
print "Done. Time: %f" % (et-st)
