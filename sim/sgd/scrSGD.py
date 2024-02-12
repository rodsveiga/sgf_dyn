import numpy as np
import itertools
from multiprocessing.pool import Pool
import os
import pickle
from trainSGD import trainSGD
##### Fix a seed for reproducibility
np.random.seed(0)

def xlog_scale(log_x_max, scale, log_base= 10): 
    '''Logaritmic scale up to log_alpha_max'''
    bd_block = np.arange(0, log_base**2, log_base) + log_base
    bd_block = bd_block[0:-1]
    xlog = np.tile(bd_block, log_x_max)
    xlog[(log_base-1) : 2*(log_base-1)] = log_base*xlog[(log_base-1) : 2*(log_base-1)]
    for j in range(1, log_x_max - 1):
        xlog[(j+1)*(log_base-1) : (j+2)*(log_base-1)] = log_base*xlog[  j*(log_base-1) :  (j+1)*(log_base-1)  ]
    xlog = np.insert(xlog, 0,  np.arange(1,log_base), axis=0)
    xlog = np.insert(xlog, len(xlog),log_base**(log_x_max+1), axis=0)
    jlog = (xlog*scale).astype(int)
    return jlog

##### p's input
##### intI and intF controls the set of p's to be run
np_points = 50
###
intI = 0
intF = 7
###
#intI = 7
#intF = 14
###
#intI = 14
#intF = 21
###
#intI = 21
#intF = 28
##### Parameters
# Data dimension
d = 1000
# Number of data points
n = 400
# Number of S random instances
nS = 1000
# Signal-to-noise ratio
snr = 1./5.
# Max number of (S)GD iterations
n_it_max= int(1e6)  
##### Number of SGD runs 
n_SGD = 1
###########################
##### Store only the terminal point in the dynamics
only_end = False
###########################
##### Learning rate
lr = 1./d
##### Data matrix
X = np.random.randn(n,d)
##### Beta star d-dimension (ground true)
beta_star_ = np.random.randn(d)
beta_star = np.copy(beta_star_) / np.linalg.norm(beta_star_)
###### Initializing the estimator
beta0_ = np.random.randn(d)
beta0  = np.copy(beta0_)/np.linalg.norm(beta0_)
##### Targets
y = X @ beta_star + snr*np.random.randn(n)
#### Plot
log_x_max = (np.log10(n_it_max)-1).astype(int)  
plot_list = xlog_scale(log_x_max, scale=1., log_base= 10)
if only_end:
    plot_list = plot_list[-1:]
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
pID = '{:02d}'.format(intI) + '{:02d}'.format(intF) 
#### List of nS
nS_ = np.arange(0,nS)
#### Folder results
subfolder = 'lr%.0e_d%d_n%d_snr%.0e_nS%d_nsgd%d_Nmax%.0e_np%d' % (lr, d, n, snr, nS, n_SGD, n_it_max, np_points)
folder = 'results/SGD/' + subfolder
#### If the folder does not exist, create
isExist = os.path.exists(folder)
if not isExist:
    os.makedirs(folder)
path = folder + '/'

print('##### Parameters #####')
print('lr = %.0e' % lr)
print('d = %d' % d)
print('n = %d' % n)
print('snr = %.0e' % snr)
print('nS = %d' % nS)
print('n_SGD = %d' % n_SGD)
print('n_it_max = %.0e' % n_it_max)
print('np_points = %d' % np_points)
print('p = %s' % p_)
print('pID = %s' % pID)

def training(p,nS):
    np.random.seed(nS)
    return trainSGD(p, nS, lr=lr, X=X, y=y, beta0= beta0, plot_list=plot_list, n=n, d=d, n_it_max=n_it_max, n_SGD= n_SGD)

if __name__ == '__main__':
    with Pool() as pool:
        # Create an iterator p x nS
        iter = list(itertools.product(p_, nS_))
        # Run in parallel
        result = pool.starmap(training, iter)

print('   ')
print('##### Saving #####')
print('   ')

plen = len(p_)
nSlen = len(nS_)

count_aux= 0
for k in np.arange(0,nSlen*plen, nSlen):
    with open(path+'p%d_DYN_betaS.npy' % p_[count_aux], 'wb') as f:
        np.save(f, np.array(result[k:k+nSlen]))
    Sp_ = []
    for nS0 in nS_:
        np.random.seed(nS0)
        Sp_.append(np.random.choice(d, size=p_[count_aux], replace=False))
    with open(path+'p%d_S.npy' % p_[count_aux], 'wb') as f:
        np.save(f, np.array(Sp_))
    count_aux += 1


plot_list0 = np.insert(plot_list, 0, 0)
################
folder2 = path + 'data'
isExist = os.path.exists(folder2)
if not isExist:
    os.makedirs(folder2 )
path2 = folder2 + '/'
################
with open(path2+'beta0.npy', 'wb') as f:
    np.save(f, beta0)   
with open(path2+'betastar.npy', 'wb') as f:
    np.save(f, beta_star)
with open(path2+'plotlist.npy', 'wb') as f:
    np.save(f, plot_list0)
with open(path2+'X.npy', 'wb') as f:
    np.save(f, X)
with open(path2+'y.npy', 'wb') as f:
    np.save(f, y)
with open(path2+'plist.npy', 'wb') as f:
    np.save(f, p_)
with open(path2+'__parameters.txt', 'w') as f:
    f.write('lr = %.0e \n' % lr)
    f.write('d = %d \n' % d)
    f.write('n = %d \n' % n)
    f.write('snr = %.0e \n' % snr)
    f.write('nS = %d \n' % nS)
    f.write('n_SGD = %d \n' % n_SGD)
    f.write('n_it_max = %.0e \n' % n_it_max)
    f.write('np_points = %d \n' % np_points)
    f.write('p = %s \n' % p_)
    f.write('pID = %s \n' % pID)

print('   ')
print('##### END #####')
print('   ')