# Stochastic Gradient Flow Dynamics of Test Risk and its Exact Solution for Weak Features

## Description

Repository for the paper [*Stochastic Gradient Flow Dynamics of Test Risk and its Exact Solution for Weak Features*](https://arxiv.org/abs/2202.00293). 

<p float="center">
  <img src="https://github.com/rodsveiga/sgf_dyn/blob/main/figures/fig04_image.jpg" height="350">
</p>


## Prerequisites
- [python](https://www.python.org/) >= 3.6
- [cython](https://cython.readthedocs.io/en/latest/#)

## Structure

In this repository we provide the code and some guided examples to help the reader to reproduce the figures. The repository is structured as follows.

| Folder ```/sim``` (simulations) | Description                                                                           |
|---------------------------------|---------------------------------------------------------------------------------------|
| ```/gd```                       | ```scrGD.py```: script to train GD importing Cython code from ```trainGD.pyx```       |
| ```/sgd```                      | ```scrSGD.py```: script to train SGD importing Cython code from ```trainSGD.pyx```    |
| ```/compute_eg```               | ```scrSGD.py```: script to compute EG importing Cython code from ```compute_eg.pyx``` |                         

The notebooks `how_to.ipynb` inside each subfolder are intended to be self-explanatory.

| Folder ```/theory``` (theoretical results) | Description                                                                           |
|--------------------------------------------|---------------------------------------------------------------------------------------|
| ```/gd```                                  | ```scrGD.py```: A       |
| ```/sgd```                                 | ```scrSGD.py```: B    |
| ```/compute_eg```                          | ```scrSGD.py```: C |   

## Building cython code

The folder ```/sim``` and ```/ode``` use cython code. To build, run `python setup.py build_ext --inplace` on the respective folder. Then simply start a python session and do whether `from sim import sim` or `from ode import ode` and use the imported function as described in the `how_to.ipynb` notebooks.

## Reference

- *Stochastic Gradient Flow Dynamics of Test Risk and its Exact Solution for Weak Features*; Rodrigo Veiga, Anastasia Remizova and Nicolas Macris; [arXiv:2202.00293](https://arxiv.org/abs/2202.00293) [stat.ML]

