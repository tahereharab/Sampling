#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:24:56 2019

@author: tahereh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:31:13 2019

@author: tahereh
"""


import time
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian
import math
from scipy.stats import entropy



def objective(x):
    function = 0 #cosine
    
    #pittsburgh
    function = x[0] * (x[21] * 0.23) + x[1] * (x[21] * 0.16) + x[2] * (x[21] * 0.09) + x[3] * (x[21] * 0.11) + x[4] * (x[21] * 0.12) + x[5] * (x[21] * 0.09) + x[6] * (x[22] * 0.23) + x[7] * (x[22] * 0.16) + x[8] * (x[22] * 0.09) + x[9] * (x[22] * 0.11) + x[10] * (x[22] * 0.12) + x[11] * (x[22] * 0.09) + x[12] * (x[23] * 0.23) + x[13] * (x[23] * 0.16) + x[14] * (x[23] * 0.09) + x[15] * (x[23] * 0.11) + x[16] * (x[23] * 0.12) + x[17] * (x[23] * 0.09) + x[18] * (x[21] * 0.2) + x[19] * (x[22] * 0.2) + x[20] * (x[23] * 0.2)
    function = function / math.sqrt((x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2) 
    * ((x[21] * 0.23)**2 + (x[21] * 0.16)**2 + (x[21] * 0.09)**2 + (x[21] * 0.11)**2 + (x[21] * 0.12)**2 + (x[21] * 0.09)**2 + (x[22] * 0.23)**2 + (x[22] * 0.16)**2 + (x[22] * 0.09)**2 + (x[22] * 0.11)**2 + (x[22] * 0.12)**2 + (x[22] * 0.09)**2 + (x[23] * 0.23)**2 + (x[23] * 0.16)**2 + (x[23] * 0.09)**2 + (x[23] * 0.11)**2 + (x[23] * 0.12)**2 + (x[23] * 0.09)**2 + (x[21] * 0.2)**2 + (x[22] * 0.2)**2 + (x[23] * 0.2)**2))
    
    #----------------------
    #new york
    """
    function = x[0] * (x[21] * 0.16) + x[1] * (x[21] * 0.16) + x[2] * (x[21] * 0.13) + x[3] * (x[21] * 0.12) + x[4] * (x[21] * 0.1) + x[5] * (x[21] * 0.1) + x[6] * (x[22] * 0.16) + x[7] * (x[22] * 0.16) + x[8] * (x[22] * 0.13) + x[9] * (x[22] * 0.12) + x[10] * (x[22] * 0.1) + x[11] * (x[22] * 0.1) + x[12] * (x[23] * 0.16) + x[13] * (x[23] * 0.16) + x[14] * (x[23] * 0.13) + x[15] * (x[23] * 0.12) + x[16] * (x[23] * 0.1) + x[17] * (x[23] * 0.1) + x[18] * (x[21] * 0.23) + x[19] * (x[22] * 0.23) + x[20] * (x[23] * 0.23)
    function = function / math.sqrt((x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2) 
    * ((x[21] * 0.16)**2 + (x[21] * 0.16)**2 + (x[21] * 0.13)**2 + (x[21] * 0.12)**2 + (x[21] * 0.1)**2 + (x[21] * 0.1)**2 + (x[22] * 0.16)**2 + (x[22] * 0.16)**2 + (x[22] * 0.13)**2 + (x[22] * 0.12)**2 + (x[22] * 0.1)**2 + (x[22] * 0.1)**2 + (x[23] * 0.16)**2 + (x[23] * 0.16)**2 + (x[23] * 0.13)**2 + (x[23] * 0.12)**2 + (x[23] * 0.1)**2 + (x[23] * 0.1)**2 + (x[21] * 0.23)**2 + (x[22] * 0.23)**2 + (x[23] * 0.23)**2))
    """
    #----------------------
    #florida
    """
    function = x[0] * (x[21] * 0.13) + x[1] * (x[21] * 0.13) + x[2] * (x[21] * 0.12) + x[3] * (x[21] * 0.13) + x[4] * (x[21] * 0.13) + x[5] * (x[21] * 0.13) + x[6] * (x[22] * 0.13) + x[7] * (x[22] * 0.13) + x[8] * (x[22] * 0.12) + x[9] * (x[22] * 0.13) + x[10] * (x[22] * 0.13) + x[11] * (x[22] * 0.13) + x[12] * (x[23] * 0.13) + x[13] * (x[23] * 0.13) + x[14] * (x[23] * 0.12) + x[15] * (x[23] * 0.13) + x[16] * (x[23] * 0.13) + x[17] * (x[23] * 0.13) + x[18] * (x[21] * 0.23) + x[19] * (x[22] * 0.23) + x[20] * (x[23] * 0.23)
    function = function / math.sqrt((x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2) 
    * ((x[21] * 0.13)**2 + (x[21] * 0.13)**2 + (x[21] * 0.12)**2 + (x[21] * 0.13)**2 + (x[21] * 0.13)**2 + (x[21] * 0.13)**2 + (x[22] * 0.13)**2 + (x[22] * 0.13)**2 + (x[22] * 0.12)**2 + (x[22] * 0.13)**2 + (x[22] * 0.13)**2 + (x[22] * 0.13)**2 + (x[23] * 0.13)**2 + (x[23] * 0.13)**2 + (x[23] * 0.12)**2 + (x[23] * 0.13)**2 + (x[23] * 0.13)**2 + (x[23] * 0.13)**2 + (x[21] * 0.23)**2 + (x[22] * 0.23)**2 + (x[23] * 0.23)**2))
    """
    #----------------------
    #college park
    """
    function = x[0] * (x[21] * 0.35) + x[1] * (x[21] * 0.08) + x[2] * (x[21] * 0.07) + x[3] * (x[21] * 0.08) + x[4] * (x[21] * 0.04) + x[5] * (x[21] * 0.03) + x[6] * (x[22] * 0.35) + x[7] * (x[22] * 0.08) + x[8] * (x[22] * 0.07) + x[9] * (x[22] * 0.08) + x[10] * (x[22] * 0.04) + x[11] * (x[22] * 0.03) + x[12] * (x[23] * 0.35) + x[13] * (x[23] * 0.08) + x[14] * (x[23] * 0.07) + x[15] * (x[23] * 0.08) + x[16] * (x[23] * 0.04) + x[17] * (x[23] * 0.03) + x[18] * (x[21] * 0.35) + x[19] * (x[22] * 0.35) + x[20] * (x[23] * 0.35)
    function = function / math.sqrt((x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2) 
    * ((x[21] * 0.35)**2 + (x[21] * 0.08)**2 + (x[21] * 0.07)**2 + (x[21] * 0.08)**2 + (x[21] * 0.04)**2 + (x[21] * 0.03)**2 + (x[22] * 0.35)**2 + (x[22] * 0.08)**2 + (x[22] * 0.07)**2 + (x[22] * 0.08)**2 + (x[22] * 0.04)**2 + (x[22] * 0.03)**2 + (x[23] * 0.35)**2 + (x[23] * 0.08)**2 + (x[23] * 0.07)**2 + (x[23] * 0.08)**2 + (x[23] * 0.04)**2 + (x[23] * 0.03)**2 + (x[21] * 0.35)**2 + (x[22] * 0.35)**2 + (x[23] * 0.35)**2))
    """
    
    function = 1 - function 
    return function
 
   
def fun_der(x):
    return Jacobian(lambda x: objective(x))(x).ravel()


def fun_hess(x):
    return Hessian(lambda x: objective(x))(x)


def constraint42(x):
    sum_eq = 1.0
    for i in range(0,21):
        sum_eq = sum_eq - x[i]   
    return sum_eq
    

def constraint43(x):
    sum_eq = 1.0
    for i in range(21,24):
        sum_eq = sum_eq - x[i]
    return sum_eq

# added as a new constraint for #female = 2*#male
def constraint44(x):
    sum_eq = 0
    sum_eq = sum_eq + x[22] - 2 * x[21]
    return sum_eq
    
"""
def constraint44(x):
    sum_eq = 1.0
    for i in range(24,31):
        sum_eq = sum_eq - x[i]
    return sum_eq
"""   
    
def shannonEntropy(final):
    pk = []
    for f in final:
        pk.append(f/100)
        
    ent = entropy(pk)
    return ent       
    
if __name__ == "__main__": 
    
    start_time = time.time()
    initial = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
               ,0.0,0.0,0.0,0.0,0.0,0.000001,0.0,0.48, 0.48, 0.0]
    
    #optimize
    Nout = 100
    
    bx1= (0,16/Nout)
    bx2= (0, 21/Nout)
    bx3= (0, 42/Nout)
    bx4= (0, 55/Nout)
    bx5= (0, 83/Nout)
    bx6= (0 , 49/Nout)
    bx7= (0, 35/Nout)
    bx8= (0, 72/Nout)
    bx9= (0, 109/Nout)
    bx10= (0, 151/Nout)
    bx11= (0, 179/Nout)
    bx12= (0, 63/Nout)
    #bx13= (0,0)
    bx13= (0,1/Nout)
    #bx14= (0,0)
    bx14= (0,1/Nout)
    bx15= (0,0)
    bx16= (0,0)
    #bx17= (0,0)
    bx17= (0,5/Nout)
    bx18= (0,0)
    bx19= (0,3/Nout)
    bx20= (0,4/Nout)
    bx21= (0,0)
    # added as a new constraint for #female = 2*#male
    bg1 = (0, 1)
    bg2 = (0, 1)
    bg3 = (0, 0.02)
    bnds = (bx1, bx2, bx3, bx4, bx5 , bx6, bx7, bx8, 
            bx9, bx10, bx11, bx12, bx13, bx14, bx15, bx16, bx17, bx18, 
            bx19, bx20, bx21 , bg1, bg2, bg3)
    #---------------
    con42 = {'type': 'eq', 'fun': constraint42}
    con43 = {'type': 'eq', 'fun': constraint43}
    # added as a new constraint for female = 2*male
    con44 = {'type': 'eq', 'fun': constraint44}
    
    cons = ([con42,con43, con44]) 
    #---------------
    solution = minimize(objective,initial,method='SLSQP', jac = None, hess= None, bounds=bnds,constraints=cons, options={'disp':True})
    #solution = minimize(objective, initial, method='dogleg', jac=fun_der, hess=fun_hess, bounds=bnds,constraints=cons)
    x = solution.x
    #---------------
    # show final objective
    print('Final SSE Objective: ' + str(objective(x)))
    print("--- %s seconds ---" % (time.time() - start_time))
    # print solution
    print('Solution')
    final = []
    for i in range(0,21):
        print('x' , i ,  ' = ' + str(round(x[i],5)))
        final.append(x[i])
        
    for i in range(21,24):
        print('gender' , i ,  ' = ' + str(round(x[i],5))) 

    
    #entro = shannonEntropy(final)
    #print('entropy = ', entro)
    #print(solution.grad)




