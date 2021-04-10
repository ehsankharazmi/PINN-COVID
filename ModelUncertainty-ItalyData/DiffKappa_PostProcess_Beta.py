# -*- coding: utf-8 -*-
"""
 

"""


import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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
import pandas

# np.random.seed(1234)
# tf.set_random_seed(1234)
# tf.random.set_seed(1234)

#%%
#read results  
S_pred_total = [] 
I_pred_total = [] 
D_pred_total = [] 
R_pred_total = []  
Kappa1_pred_total = [] 
Kappa2_pred_total = [] 
Kappa3_pred_total = [] 
Kappa4_pred_total = []  
Beta_pred_total = []  


from datetime import datetime
now = datetime.now()
# dt_string = now.strftime("%m-%d-%H-%M")
# dt_string = now.strftime("%m-%d")
dt_string = '03-19'


# for j in [1,2,3,4,5,6,7,8,9,10]:
for j in np.arange(1,5,1):

    casenumber ='set' +str(j)
    current_directory = os.getcwd()
    relative_path_results = '/SIRD-DiffKappa-Beta/Train-Results-'+dt_string+'-'+casenumber+'/'
    read_results_to = current_directory + relative_path_results 
    
    S_pred = np.loadtxt(read_results_to + 'S.txt') 
    I_pred = np.loadtxt(read_results_to + 'I.txt') 
    D_pred = np.loadtxt(read_results_to + 'D.txt') 
    R_pred = np.loadtxt(read_results_to + 'R.txt') 
    Kappa1_pred = np.loadtxt(read_results_to + 'Kappa1.txt')
    Kappa2_pred = np.loadtxt(read_results_to + 'Kappa2.txt')
    Kappa3_pred = np.loadtxt(read_results_to + 'Kappa3.txt')
    Kappa4_pred = np.loadtxt(read_results_to + 'Kappa4.txt') 
    Beta_pred = np.loadtxt(read_results_to + 'Beta.txt') 

    S_pred_total.append(S_pred) 
    I_pred_total.append(I_pred) 
    D_pred_total.append(D_pred) 
    R_pred_total.append(R_pred)  
    Kappa1_pred_total.append(Kappa1_pred) 
    Kappa2_pred_total.append(Kappa2_pred) 
    Kappa3_pred_total.append(Kappa3_pred) 
    Kappa4_pred_total.append(Kappa4_pred)  
    Beta_pred_total.append(Beta_pred)  
    
#%%
#Average  
S_pred_mean = np.mean(S_pred_total, axis=0) 
I_pred_mean = np.mean(I_pred_total, axis=0) 
D_pred_mean = np.mean(D_pred_total, axis=0) 
R_pred_mean = np.mean(R_pred_total, axis=0)   
Kappa1_pred_mean = np.mean(Kappa1_pred_total, axis=0)
Kappa2_pred_mean = np.mean(Kappa2_pred_total, axis=0)
Kappa3_pred_mean = np.mean(Kappa3_pred_total, axis=0)
Kappa4_pred_mean = np.mean(Kappa4_pred_total, axis=0) 
Beta_pred_mean = np.mean(Beta_pred_total, axis=0) 
 
S_pred_std = np.std(S_pred_total, axis=0) 
I_pred_std = np.std(I_pred_total, axis=0) 
D_pred_std = np.std(D_pred_total, axis=0) 
R_pred_std = np.std(R_pred_total, axis=0)  
Kappa1_pred_std = np.std(Kappa1_pred_total, axis=0) 
Kappa2_pred_std = np.std(Kappa2_pred_total, axis=0) 
Kappa3_pred_std = np.std(Kappa3_pred_total, axis=0)
Kappa4_pred_std = np.std(Kappa4_pred_total, axis=0)  
Beta_pred_std = np.std(Beta_pred_total, axis=0)  

#%%
#save results  
current_directory = os.getcwd()
relative_path_results = '/SIRD-DiffKappa-Beta/Train-Results-'+dt_string+'-Average/'
save_results_to = current_directory + relative_path_results 
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

np.savetxt(save_results_to + 'S_pred_mean.txt', S_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'I_pred_mean.txt', I_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'D_pred_mean.txt', D_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'R_pred_mean.txt', R_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'Kappa1_pred_mean.txt', Kappa1_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa2_pred_mean.txt', Kappa2_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa3_pred_mean.txt', Kappa3_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa4_pred_mean.txt', Kappa4_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Beta_pred_mean.txt', Beta_pred_mean.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_pred_std.txt', S_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'I_pred_std.txt', I_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'D_pred_std.txt', D_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'R_pred_std.txt', R_pred_std.reshape((-1,1)))    
np.savetxt(save_results_to + 'Kappa1_pred_std.txt', Kappa1_pred_std.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa2_pred_std.txt', Kappa2_pred_std.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa3_pred_std.txt', Kappa3_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'Kappa4_pred_std.txt', Kappa4_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'Beta_pred_std.txt', Beta_pred_std.reshape((-1,1))) 

