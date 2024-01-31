# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
import time

cpdef trainSGD(int p, int nS, double lr, double[:,:] X, double[:] y, double[:] beta0, np.int64_t[:] plot_list, int n, int d, int n_it_max, int n_SGD):
    cdef unsigned int k, r, s, q
    #####
    cdef double *arr = <double*>malloc(p * sizeof(double))
    cdef double[:] betaS = <double[:p]>arr
    #####
    cdef double *arr5 = <double*>malloc(p * sizeof(double))
    cdef double[:] betaS0 = <double[:p]>arr5
    cdef long *arr2 = <long*>malloc(p * sizeof(double))
    cdef long[:] S = <long[:p]>arr2
    #####
    cdef double *arr6 = <double*>malloc( n * p * sizeof(double))
    cdef double[:,:] XS = <double[:n, :p]>arr6

    print('Beginning SGD training ----- p= %d --- nS= %d' % (p,nS))
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
       
    ######## STOCHASTIC GRADIENT DESCENT
    ##############
    start = time.time()
    ##############

    betaS_av = np.zeros((len(plot_list)+1,p))
    betaS_av[0] = np.array(betaS0)

    for N in range(n_SGD):
        ### N-th SGD run
        betaS = np.copy(betaS0)
        #######
        count_aux = 1
        for k in range(1,n_it_max+1):
            ## Randomly select a training sample
            jk = np.random.randint(n)
            ## Inner product: XS_jk and beta_k
            sgd_aux0 = 0.
            for r in range(0,p):
                sgd_aux0 += XS[jk,r]*betaS[r]
            Djk = y[jk] - sgd_aux0
            ## SGD update
            for s in range(0,p):  
                betaS[s] += lr*Djk*XS[jk,s]

            ## Retrieving dynamics
            save_k = k in plot_list
            if save_k:
                betaS_av[count_aux] += np.array(betaS)/n_SGD
                count_aux += 1
    ##############
    end = time.time()
    ##############
    free(arr6)
    free(arr5)
    free(arr)

    print('Time elapsed SGD: %f' % (end-start))
    print('End ----- p= %d --- nS= %d' % (p,nS))
    print(' ')

    return betaS_av

