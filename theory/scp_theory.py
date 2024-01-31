import numpy as np
import pandas as pd
from multiprocessing.pool import Pool
from theory import get_z_covariation
from collections import defaultdict
import pickle
##### Fix a seed for reproducibility
np.random.seed(0)

#n = 40
#d = 100
#######
n = 400
d = 1000
#######
psi = d / n
l2 = 0.04

lr = 1./d 
n_t_points = 1003
t_ = np.logspace(0, 6, n_t_points)
t1 = np.arange(20,100, 10)
t2 = np.arange(20,100, 10)
t3 = np.arange(200,1000, 100)
t4 = np.arange(2000,10000, 1000)
t5 = np.arange(20000,100000, 10000)
t6 = np.arange(200000,1000000, 100000)
args = (t1, t2, t3, t4, t5, t6)
xt = np.concatenate(args)

t = np.unique(np.sort(np.concatenate([t_, xt])))
betadiff2 = 2.0

p_ = np.arange(1, d, 1)
alpha_ = p_ / n


clnames = defaultdict(list)
for l,c in zip(t, np.arange(len(t))):
    clnames[l].append(c)
clnames = dict(clnames)


def cov(alpha):
    return get_z_covariation(alpha, psi, l2, t * lr, betadiff2)


if __name__ == '__main__':
    with Pool() as pool:
        # Run in parallel
        result = pool.map(cov, alpha_)


zcov_ = np.array(result).reshape(len(p_), t.shape[0])
cov_df = pd.DataFrame(zcov_,index=alpha_, columns=clnames).rename_axis(index='alpha', columns="time")
cov_df.to_pickle('theory_beta2_scale_d1000.pkl') 