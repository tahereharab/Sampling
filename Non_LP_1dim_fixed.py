#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:49:21 2020

@author: tahereh
"""

import pandas as pd
import numpy as np
import time
import math
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian
from scipy.stats import entropy

def objective(x):
    
    function = 0
    function = x[0] * (0.2) + x[1] * (0.23) + x[2] * (0.16) + x[3] * (0.09) + x[4] * (0.11) + x[5] * (0.12) + x[6] * (0.09) 
    function = function / math.sqrt((x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2) 
    * ((0.2)**2 + (0.23)**2 + (0.16)**2 + (0.09)**2 + (0.11)**2 + (0.12)**2 + (0.09)**2))
    function = 1 - function
    return function
    
def constraint(x):
    sum_eq = 1.0
    for i in range(0,7):
        sum_eq = sum_eq - x[i]   
    return sum_eq
         
if __name__ == "__main__": 
    
    start_time = time.time()
    initial = [0.0,0.0,0.0,0.0,0.0,0.0,0.0000001]
    #---------------
    #optimize
    Nout = 100
    bx1= (0,7/Nout)
    bx2= (0, 52/Nout)
    bx3= (0, 94/Nout)
    bx4= (0, 151/Nout)
    bx5= (0, 206/Nout)
    bx6= (0 , 267/Nout)
    bx7= (0, 112/Nout)
    bnds = (bx1, bx2, bx3, bx4, bx5 , bx6, bx7)
    #---------------
    con42 = {'type': 'eq', 'fun': constraint}
    cons = ([con42]) 
    #---------------
    solution = minimize(objective,initial,method='SLSQP', jac = None, hess= None, bounds=bnds,constraints=cons, options={'disp':True})
    x = solution.x
    #---------------
    # show final objective
    print('Final SSE Objective: ' + str(objective(x)))
    print("--- %s seconds ---" % (time.time() - start_time))
    # print solution
    print('Solution')
    for i in range(0,7):
        print('x' , i ,  ' = ' + str(round(x[i],5)))
    #entro = shannonEntropy(x)
    #print('entropy = ', entro)


