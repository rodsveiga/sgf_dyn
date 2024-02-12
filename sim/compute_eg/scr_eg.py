import numpy as np
import time 
import pickle
import pandas as pd
import os
from multiprocessing.pool import Pool
from compute_eg import eg
from collections import defaultdict


root = '/home/rveiga/simul_ib/gmodel_term3/results/'
#root = '/localhome/rveiga/Dropbox/Research/trained_models/'

ntest = 1e5
n_SGD = 1

d = 1000 
n = 400  
nS = 1000
n_SGD = 1
lr = 1./d
snr = 1./5.
n_it_max = 1e+6 
np_points = 50 
only_last = False
#######################
intI = 0
intF = 28

#### List of p's
if np.abs(d-1000) > 0:
    d_AUX = 1000
    np_points_AUX = 50
    n_AUX = 400
    p__AUX = np.arange(0, d_AUX , int(d_AUX/np_points_AUX))
    p__AUX[0] = 1
    p_AUX = p__AUX[intI:intF]
    alpha_AUX = p_AUX/n_AUX 
    p_ = (alpha_AUX*n).astype(int)
    p_[0] = 1
else:
    p__ = np.arange(0, d, int(d/np_points))
    p__[0] = 1
    p_ = p__[intI:intF]



print('##### Parameters #####')
print('ntest = %.0e' % ntest)
print('lr = %.0e' % lr)
print('d = %d' % d)
print('n = %d' % n)
print('snr = %.0e' % snr)
print('nS = %d' % nS)
print('n_it_max = %.0e' % n_it_max)
print('np_points = %d' % np_points)
print('p = %s' % p_)


# For GD:  set gd = 1                                                               
# For SGD: set gd = 0
gd  = 1


if gd > 0:
    folder =  root + 'GD/lr%.0e_d%d_n%d_snr%.0e_nS%d_Nmax%.0e_np%d/' % (lr, d, n, snr, nS, n_it_max, np_points)
else:
    folder = root + 'SGD/lr%.0e_d%d_n%d_snr%.0e_nS%d_nsgd%d_Nmax%.0e_np%d/' % (lr, d, n, snr, nS, n_SGD, n_it_max, np_points)


eg_folder = folder + 'eg'
#### If the folder does not exist, create
isExist = os.path.exists(eg_folder)
if not isExist:
    os.makedirs(eg_folder)



def EGcomp(p, folder, eg_folder , snr, ntest, only_last):
    print('---- Starting %d' % p)
    start = time.time()
    ###
    with open(folder+'p%d_DYN_betaS.npy' % p, 'rb') as f:
         betaSp = np.load(f)

    with open(folder+'p%d_S.npy' % p, 'rb') as f:
         Sp = np.load(f)

    with open(folder+'data/'+'betastar.npy', 'rb') as f:
         beta_star = np.load(f)

    with open(folder+'data/'+'plotlist.npy', 'rb') as f:
         plotlist = np.load(f)

    with open(folder+'data/'+'plist.npy', 'rb') as f:
         plist = np.load(f)
    #######
    word = 'DYN_'
    index_array = np.arange(len(plotlist))
    if only_last:
        plotlist = np.array([plotlist[0],plotlist[-1]])
        index_array = np.array([0,-1])
        word = 'IF_'
    ######
    clnames = defaultdict(list)
    for l,c in zip(plotlist, np.arange(len(plotlist))):
        clnames[l].append(c)
    clnames = dict(clnames)
    #############################    
    erS_ = []
    for s in range(nS):
        bS = betaSp[s]
        #print(bS.shape)
        S = Sp[s]
        erSt_ = []
        for t in index_array:
            bt = bS[t]
            error = eg(bt, beta_star, snr, S, ntest)
            erSt_.append(error)
        erS_.append(erSt_)

    df_p = pd.DataFrame(np.array(erS_),index=np.arange(nS), columns=clnames).rename_axis(index='S', columns="time")
    path = eg_folder+ '/' + word + 'eg_ntest%.0e_p%d' % (ntest, p) + '.pkl'
    df_p.to_pickle(path)
    end = time.time()
    print('Elapsed time %.3f' % (end - start))
    print('---- Finishing %d' % p)



def compEG_(p):
    return EGcomp(p, folder=folder, eg_folder=eg_folder, snr=snr, ntest=ntest, only_last=only_last)


if __name__ == '__main__':
    with Pool() as pool:
        # Run in parallel
        result = pool.map(compEG_, p_)