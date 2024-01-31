# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
import time

cpdef trainGD(int p, int nS, double lr, double[:,:] X, double[:] y, double[:] beta0, np.int64_t[:] plot_list, int n, int d, int n_it_max):
    cdef unsigned int k, r, s, u, q
    #####
    cdef double *arr = <double*>malloc(p * sizeof(double))
    cdef double[:] betaS = <double[:p]>arr
    #####
    cdef double *arr3 = <double*>malloc(n * sizeof(double))
    cdef double[:] D = <double[:n]>arr3
    #####
    cdef double *arr4 = <double*>malloc(p * sizeof(double))
    cdef double[:] GRAD = <double[:p]>arr4
    #####
    cdef double *arr5 = <double*>malloc(p * sizeof(double))
    cdef double[:] betaS0 = <double[:p]>arr5
    #####
    cdef double gd_aux0, gd_aux1
    #####
    cdef long *arr2 = <long*>malloc(p * sizeof(double))
    cdef long[:] S = <long[:p]>arr2
    #####
    cdef double *arr6 = <double*>malloc( n * p * sizeof(double))
    cdef double[:,:] XS = <double[:n, :p]>arr6

    print('Beginning GD training ----- p= %d --- nS= %d' % (p,nS))
    ###### Select randomly p integers from [d]
    S  = np.random.choice(d, size=p, replace=False)
         
    ###### Initializing the estimator
    betaS0[:] = 0.
    for q in range(0,p):
        betaS0[q] = beta0[S[q]]
    del beta0

    ###### Constructing XS
    for r in range(0,n):
        for q in range(0,p):
            XS[r,q] = X[r,S[q]]
    del X
    free(arr2)
       
    ######## GRADIENT DESCENT
    #####
    betaS_ = []
    betaS_.append(np.array(betaS0))
    betaS = np.copy(betaS0)
    free(arr5)
    ##############
    start = time.time()
    ##############

    for k in range(1,n_it_max+1):
        #### GD ####
        ## Inner product: XS and beta_k 
        D[:] = 0. 
        GRAD[:] = 0.
        for r in range(0,n):
            gd_aux0 = 0.
            for s in range(0,p):
                gd_aux0 += XS[r,s]*betaS[s]
            D[r] = y[r] - gd_aux0
        ## Inner product: y - XS*betaS times xs
        for r in range(0,p):
            gd_aux1 = 0.
            for s in range(0,n):
                gd_aux1 += XS[s,r]*D[s]
            GRAD[r] = gd_aux1   
        ## GD update
        for u in range(0,p):
            betaS[u] += (lr/n)*GRAD[u]

        ## Retrieving dynamics
        save_k = k in plot_list
        if save_k:
            betaS_.append(np.array(betaS))
            
    ##############
    end = time.time()
    ##############
    free(arr3)
    free(arr4)
    free(arr6)
    free(arr)

    print('Time elapsed GD: %f' % (end-start))
    print('End ----- p= %d --- nS= %d' % (p,nS))
    print(' ')

    return np.array(betaS_)

