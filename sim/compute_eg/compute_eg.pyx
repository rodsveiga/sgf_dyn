# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


cdef inline double square(double x): return x*x

cpdef public double eg(double[:] betaS, double[:] beta_st, double snr, long [:] S, int ntest):
    cdef unsigned int r, s, u, q, d = beta_st.shape[0], p= betaS.shape[0]
    cdef double error= 0., ynew, ynew_hat
    cdef double *arr0 = <double*>malloc(d * sizeof(double))
    cdef double[:] xnew = <double[:d]>arr0
    cdef double *arr1 = <double*>malloc(p * sizeof(double))
    cdef double[:] xnewS = <double[:p]>arr1
    np.random.seed(0)
    for r in range(0,ntest):
        xnew = np.random.randn(d)
        #### Ground truth
        ynew = 0.
        for s in range(0,d):
            ynew +=  xnew[s]*beta_st[s]
        # Noise on the ground truth
        ynew += snr*np.random.randn()
        #### Estimator
        ynew_hat = 0.
        # Constructing xnewS
        for q in range(0,p):
            xnewS[q] = xnew[S[q]]
        # Computing the inner product
        for u in range(0,p):
            ynew_hat += xnewS[u]*betaS[u]    
        error += 0.5*square(ynew-ynew_hat)/ntest 
    free(arr0)
    free(arr1)
    return error