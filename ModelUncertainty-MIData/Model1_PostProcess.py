# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:18:25 2021

@author: Administrator 

"""


import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas
import math 
from math import gamma
import matplotlib.dates as mdates
import tensorflow as tf
import numpy as np
from numpy import *
# from numpy import matlib as mb
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import datetime
from pyDOE import lhs
# from scipy.special import gamma
start_time = time.time()


# np.random.seed(1234)
# tf.set_random_seed(1234)
# tf.random.set_seed(1234)

#%%
#read results  

S_pred_total = []
E_pred_total = []
I_pred_total = []
J_pred_total = []
D_pred_total = []
H_pred_total = []
R_pred_total = [] 
BetaI_pred_total = [] 
Rc_pred_total = []
p_pred_total = []
q_pred_total = [] 
I_new_pred_total = []
H_new_pred_total = []
D_new_pred_total = []
I_sum_pred_total = []
H_sum_pred_total = []
D_sum_pred_total = []

from datetime import datetime
now = datetime.now()
# dt_string = now.strftime("%m-%d-%H-%M")
dt_string = now.strftime("%m-%d")
# dt_string = '05-04'

for j in [1,2,3]:
# for j in np.arange(1,11,1):
# for j in [1]:

    casenumber ='set' +str(j)
    current_directory = os.getcwd()
    relative_path_results = '/Model1/Train-Results-'+dt_string+'-'+casenumber+'/'
    # relative_path_results = '/Model1/Train-Results-'+casenumber+'/'
    read_results_to = current_directory + relative_path_results 
    
    S_pred = np.loadtxt(read_results_to + 'S.txt')
    E_pred = np.loadtxt(read_results_to + 'E.txt')
    I_pred = np.loadtxt(read_results_to + 'I.txt')
    J_pred = np.loadtxt(read_results_to + 'J.txt')
    D_pred = np.loadtxt(read_results_to + 'D.txt')
    H_pred = np.loadtxt(read_results_to + 'H.txt')
    R_pred = np.loadtxt(read_results_to + 'R.txt')
    BetaI_pred = np.loadtxt(read_results_to + 'BetaI.txt')
    Rc_pred = np.loadtxt(read_results_to + 'Rc.txt')
    p_pred = np.loadtxt(read_results_to + 'p.txt')
    q_pred = np.loadtxt(read_results_to + 'q.txt')
    I_new_pred = np.loadtxt(read_results_to + 'I_new.txt')
    H_new_pred = np.loadtxt(read_results_to + 'H_new.txt')
    D_new_pred = np.loadtxt(read_results_to + 'D_new.txt')
    I_sum_pred = np.loadtxt(read_results_to + 'I_sum.txt')
    H_sum_pred = np.loadtxt(read_results_to + 'H_sum.txt')
    D_sum_pred = np.loadtxt(read_results_to + 'D_sum.txt') 

    S_pred_total.append(S_pred)
    E_pred_total.append(E_pred)
    I_pred_total.append(I_pred)
    J_pred_total.append(J_pred)
    D_pred_total.append(D_pred)
    H_pred_total.append(H_pred)
    R_pred_total.append(R_pred) 
    I_new_pred_total.append(I_new_pred)
    H_new_pred_total.append(H_new_pred)
    D_new_pred_total.append(D_new_pred)
    I_sum_pred_total.append(I_sum_pred)
    H_sum_pred_total.append(H_sum_pred)
    D_sum_pred_total.append(D_sum_pred)
    BetaI_pred_total.append(BetaI_pred)
    Rc_pred_total.append(Rc_pred)
    p_pred_total.append(p_pred)
    q_pred_total.append(q_pred) 
    
#%%
#Average  
S_pred_mean = np.mean(S_pred_total, axis=0)
E_pred_mean = np.mean(E_pred_total, axis=0)
I_pred_mean = np.mean(I_pred_total, axis=0)
J_pred_mean = np.mean(J_pred_total, axis=0)
D_pred_mean = np.mean(D_pred_total, axis=0)
H_pred_mean = np.mean(H_pred_total, axis=0)
R_pred_mean = np.mean(R_pred_total, axis=0) 
I_new_pred_mean = np.mean(I_new_pred_total, axis=0)
H_new_pred_mean = np.mean(H_new_pred_total, axis=0)
D_new_pred_mean = np.mean(D_new_pred_total, axis=0)
I_sum_pred_mean = np.mean(I_sum_pred_total, axis=0)
H_sum_pred_mean = np.mean(H_sum_pred_total, axis=0)
D_sum_pred_mean = np.mean(D_sum_pred_total, axis=0)
BetaI_pred_mean = np.mean(BetaI_pred_total, axis=0) 
Rc_pred_mean = np.mean(Rc_pred_total, axis=0) 
p_pred_mean = np.mean(p_pred_total, axis=0) 
q_pred_mean = np.mean(q_pred_total, axis=0) 

 
S_pred_std = np.std(S_pred_total, axis=0)
E_pred_std = np.std(E_pred_total, axis=0)
I_pred_std = np.std(I_pred_total, axis=0)
J_pred_std = np.std(J_pred_total, axis=0)
D_pred_std = np.std(D_pred_total, axis=0)
H_pred_std = np.std(H_pred_total, axis=0)
R_pred_std = np.std(R_pred_total, axis=0) 
I_new_pred_std = np.std(I_new_pred_total, axis=0)
H_new_pred_std = np.std(H_new_pred_total, axis=0)
D_new_pred_std = np.std(D_new_pred_total, axis=0)
I_sum_pred_std = np.std(I_sum_pred_total, axis=0)
H_sum_pred_std = np.std(H_sum_pred_total, axis=0)
D_sum_pred_std = np.std(D_sum_pred_total, axis=0)
BetaI_pred_std = np.std(BetaI_pred_total, axis=0) 
Rc_pred_std = np.std(Rc_pred_total, axis=0) 
p_pred_std = np.std(p_pred_total, axis=0) 
q_pred_std = np.std(q_pred_total, axis=0) 

#%%
#save results  
current_directory = os.getcwd()
relative_path_results = '/Model1/Train-Results-'+dt_string+'-Average/'
# relative_path_results = '/Model1/Train-Results-Average/'
save_results_to = current_directory + relative_path_results 
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

np.savetxt(save_results_to + 'S_pred_mean.txt', S_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_pred_mean.txt', E_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_pred_mean.txt', I_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_pred_mean.txt', J_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_pred_mean.txt', D_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_pred_mean.txt', H_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_pred_mean.txt', R_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'I_new_pred_mean.txt', I_new_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'H_new_pred_mean.txt', H_new_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_new_pred_mean.txt', D_new_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_sum_pred_mean.txt', I_sum_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'H_sum_pred_mean.txt', H_sum_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_sum_pred_mean.txt', D_sum_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'BetaI_pred_mean.txt', BetaI_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Rc_pred_mean.txt', Rc_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'p_pred_mean.txt', p_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'q_pred_mean.txt', q_pred_mean.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_pred_std.txt', S_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_pred_std.txt', E_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_pred_std.txt', I_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_pred_std.txt', J_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_pred_std.txt', D_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_pred_std.txt', H_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_pred_std.txt', R_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'I_new_pred_std.txt', I_new_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'H_new_pred_std.txt', H_new_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_new_pred_std.txt', D_new_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_sum_pred_std.txt', I_sum_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'H_sum_pred_std.txt', H_sum_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_sum_pred_std.txt', D_sum_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'BetaI_pred_std.txt', BetaI_pred_std.reshape((-1,1)))   
np.savetxt(save_results_to + 'Rc_pred_std.txt', Rc_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'p_pred_std.txt', p_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'q_pred_std.txt', q_pred_std.reshape((-1,1)))  
