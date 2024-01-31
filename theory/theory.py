import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad, nquad
from scipy.special import i1
from scipy.integrate import odeint
import math
import os


def MP_alpha_minus(alpha):
    return (1 - np.sqrt(alpha)) ** 2


def MP_alpha_plus(alpha):
    return (1 + np.sqrt(alpha)) ** 2


def _MP_pdf_expression(x, alpha, alpha_minus, alpha_plus):

    return 1 / (2 * np.pi) * np.sqrt(
        np.abs((alpha_plus - x) * (x - alpha_minus))) / (alpha * x)

    
def MP_expectation(f, alpha):
    """
    Expectation of E_{x \sim MP(\alpha)} [f(x)]
    """
    if alpha == 0:
        return f(1)
    
    alpha_minus = MP_alpha_minus(alpha)
    alpha_plus = MP_alpha_plus(alpha)

    exp_val = quad(lambda x: f(x) * _MP_pdf_expression(x, alpha, alpha_minus, alpha_plus),
                   alpha_minus, alpha_plus)[0]
    return exp_val



def Etrain_inf_time(alpha, psi, l2): 
    # no dependency on betadiff2
    res = (1 - alpha / psi + l2) * (1 - np.minimum(alpha, 1))
    return res
    
    
def Etrain_inf_time_precise(p, n, d, l2): 
    # no dependency on betadiff2
    res = (1 - p / d + l2) * (1 - np.minimum(p, n) / n)
    return res
    

def _Etest_inf_time_alpha_leq_1(alpha, psi, l2, betadiff2):
    res = (1 - alpha / psi + l2) * 1 / (1 - alpha)
    return res


def _Etest_inf_time_alpha_g_1(alpha, psi, l2, betadiff2):
    res = alpha / psi * betadiff2 * (1 - 1 / alpha) + (1 - alpha / psi + l2) * (1 + 1 / (alpha - 1))
    return res
    
    
def Etest_inf_time(alpha, psi, l2, betadiff2):
    alpha = np.asarray(alpha)
    res = np.zeros(alpha.shape)
    
    is_alpha_leq_1 = alpha <= 1 
    res[is_alpha_leq_1] = _Etest_inf_time_alpha_leq_1(alpha[is_alpha_leq_1], psi, l2, betadiff2)
    res[~is_alpha_leq_1] = _Etest_inf_time_alpha_g_1(alpha[~is_alpha_leq_1], psi, l2, betadiff2)
    return res

def Etest_inf_time_precise(p, n, d, l2, betadiff2):
    p = np.asarray(p)
    res = np.full(p.shape, np.inf)
    
    is_p_l_nm1 = p < (n - 1)
    p_l_nm1 = p[is_p_l_nm1]
    res[is_p_l_nm1] = (1 - p_l_nm1 / d + l2) * (1 + p_l_nm1 / (n - p_l_nm1 - 1))
    
    is_p_g_np1 = p > (n + 1)
    p_g_np1 = p[is_p_g_np1]
    res[is_p_g_np1] = p_g_np1 / d * betadiff2 * (1 - n / p_g_np1) + (1 - p_g_np1 / d + l2) * (1 + n / (p_g_np1 - n - 1))
    
    return res


def _Etrain_alpha_leq_1(alpha, psi, l2, t, betadiff2):
    f1 = lambda x: x * math.exp(-2 * t * x)
    f2 = lambda x: math.exp(-2 * t * x)
    res = alpha * betadiff2 / psi * MP_expectation(f1, alpha) + \
          (1 - alpha / psi + l2) * (1 - alpha + alpha * MP_expectation(f2, alpha))
    return res


def _Etrain_alpha_g_1(alpha, psi, l2, t, betadiff2):
    f1 = lambda x: x * math.exp(-2 * t * alpha * x)
    f2 = lambda x: math.exp(-2 * t * alpha * x)
    res =  betadiff2 / psi * alpha * MP_expectation(f1, 1 / alpha) + \
            (1 - alpha / psi + l2) * MP_expectation(f2, 1 / alpha)
    return res


_Etrain_alpha_leq_1 = np.vectorize(_Etrain_alpha_leq_1)
_Etrain_alpha_g_1 = np.vectorize(_Etrain_alpha_g_1)


def Etrain(alpha, psi, l2, t, betadiff2):
    if alpha <= 1:
        return _Etrain_alpha_leq_1(alpha, psi, l2, t, betadiff2)
    else:
        return _Etrain_alpha_g_1(alpha, psi, l2, t, betadiff2)
    
Etrain = np.vectorize(Etrain)
    


def _Etest_alpha_leq_1(alpha, psi, l2, t, betadiff2):
    f1 = lambda x: math.exp(-2 * t * x)
    f2 = lambda x: (1 - math.exp(- t * x)) ** 2 / x
    res = alpha * betadiff2 / psi * MP_expectation(f1, alpha) + \
          (1 - alpha / psi + l2) * (1 + alpha * MP_expectation(f2, alpha))
    return res


def _Etest_alpha_g_1(alpha, psi, l2, t, betadiff2):
    f1 = lambda x: math.exp(-2 * alpha * t * x)
    f2 = lambda x: (1 - math.exp(- alpha * t * x)) ** 2 / x
    res = betadiff2 * (alpha - 1) / psi + betadiff2 / psi * MP_expectation(f1, 1 / alpha) + \
          (1 - alpha / psi + l2) * (1 + 1 / alpha * MP_expectation(f2, 1 / alpha))
    return res


_Etest_alpha_leq_1 = np.vectorize(_Etest_alpha_leq_1)
_Etest_alpha_g_1 = np.vectorize(_Etest_alpha_g_1)
 

def Etest(alpha, psi, l2, t, betadiff2):
    if alpha <= 1:
        return _Etest_alpha_leq_1(alpha, psi, l2, t, betadiff2)
    else:
        return _Etest_alpha_g_1(alpha, psi, l2, t, betadiff2)

Etest = np.vectorize(Etest)



def expectation_MP_exp(alpha, t, eps=1e-2, max_i1_arg=100):
    res = 1 / np.sqrt(alpha) * odeint(
        lambda y, s:  np.where(
            s == 0,
            2 * np.sqrt(alpha),
            np.where(
                4 * np.sqrt(alpha) * s < max_i1_arg,
                np.exp(- 2 * (1 + alpha) * s) * i1( 4 * np.sqrt(alpha) * s) / s,
                np.exp(- 2 * (1 - np.sqrt(alpha)) ** 2 * s) / 
                (2 * np.sqrt(2 * np.pi *  np.sqrt(alpha) * s ** 3))
            )),
        0, np.hstack(([0], t)))
    return res[1:, 0]



def get_z_covariation(alpha, psi, l2, t, betadiff2, eps=1e-15):
#     alpha = np.asarray(alpha)

    t = np.asarray(t)
    
    coef1 = 8 * np.sqrt(alpha) / psi * betadiff2
    coef2 = 1 - alpha / psi + l2
    
    aminus = MP_alpha_minus(alpha)
    aplus = MP_alpha_plus(alpha)
    denom_term = aminus / (4 * np.sqrt(alpha))
    
    res = 0
    if alpha < 1.:
        res += coef2 * (1 - alpha) / 2 * expectation_MP_exp(alpha, t, eps=1e-15)
        
#     f1 = 16 * np.exp(-2 * aminus * t) / (np.pi ** 2 * np.sqrt(alpha)) * int1
#     f2 = 2 * np.exp(-2 * aminus * t) / (np.pi ** 2) * int2
    
#     res += coef1 * f1 + coef2 * f2 

    exp_coef = - 8 * np.sqrt(alpha) * t
    
    def get_int_term(exp_coef):
        f = lambda sigma, rs: (coef1 + coef2 * (1 / (denom_term + rs) + 1 / (denom_term + sigma))) * \
                (np.exp(exp_coef * sigma) - np.exp(exp_coef * rs))/(rs - sigma) * \
                                np.sqrt(sigma * (1 - sigma) * (1 - rs) * rs) 
        int3 = dblquad(f, 0, 1, lambda sigma: sigma, lambda sigma: 1)[0] 
        return int3
    get_int_term = np.vectorize(get_int_term)
    
    
    res += 2 * np.exp(-2 * aminus * t) / (np.pi ** 2) * get_int_term(exp_coef)
    
    res *= alpha
    return res/2.
