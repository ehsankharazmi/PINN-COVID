# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:42:31 2021

@author: Administrator 

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

#Load Data 

current_directory = os.getcwd()
relative_path_loadSIR = '/../Model-SIHDR-DATA/'
read_results_to = current_directory + relative_path_loadSIR 
S_SIHDR = np.loadtxt(read_results_to + 'S_pred_mean.txt')[:,None] 
I_SIHDR = np.loadtxt(read_results_to + 'I_pred_mean.txt')[:,None] 
H_SIHDR = np.loadtxt(read_results_to + 'H_pred_mean.txt')[:,None] 
D_SIHDR = np.loadtxt(read_results_to + 'D_pred_mean.txt')[:,None] 
R_SIHDR = np.loadtxt(read_results_to + 'R_pred_mean.txt')[:,None] 
S_SIHDR = S_SIHDR/1e-4
I_SIHDR = I_SIHDR/1e-4
H_SIHDR = H_SIHDR/1e-4
D_SIHDR = D_SIHDR/1e-4
R_SIHDR = R_SIHDR/1e-4


data_frame = pandas.read_csv('../Data/data-by-day.csv')  
I_new_star = data_frame['CASE_COUNT'] #T x 1 array 
H_new_star = data_frame['HOSPITALIZED_COUNT'] #T x 1 array
D_new_star = data_frame['DEATH_COUNT'] #T x 1 array 
t_data = data_frame['date_of_interest']

#7 days average 
I_new_star = I_new_star.rolling(window=7).mean()
H_new_star = H_new_star.rolling(window=7).mean()
D_new_star = D_new_star.rolling(window=7).mean()
I_new_star = I_new_star.to_numpy(dtype=np.float64)
H_new_star = H_new_star.to_numpy(dtype=np.float64)
D_new_star = D_new_star.to_numpy(dtype=np.float64)
I_new_star = I_new_star[6:]
H_new_star = H_new_star[6:]
D_new_star = D_new_star[6:]
I_new_star = I_new_star.reshape([len(I_new_star), 1])
H_new_star = H_new_star.reshape([len(H_new_star), 1])
D_new_star = D_new_star.reshape([len(D_new_star), 1]) 
I_sum_star = np.cumsum(I_new_star)
H_sum_star = np.cumsum(H_new_star)
D_sum_star = np.cumsum(D_new_star)
I_sum_star = I_sum_star.reshape([len(I_sum_star), 1])
H_sum_star = H_sum_star.reshape([len(H_sum_star), 1])
D_sum_star = D_sum_star.reshape([len(D_sum_star), 1])


I_new_train = I_new_star 
D_new_train = D_new_star 
H_new_train = H_new_star
I_sum_train = I_sum_star 
D_sum_train = D_sum_star 
H_sum_train = H_sum_star 



#%%
#read results  
S_pred_total = [] 
I_pred_total = [] 
D_pred_total = []
H_pred_total = []
R_pred_total = [] 
BetaI_pred_total = []
# Kappa_pred_total = []
p_pred_total = []
q_pred_total = [] 
I_new_pred_total = []
H_new_pred_total = []
D_new_pred_total = []
I_sum_pred_total = []
H_sum_pred_total = []
D_sum_pred_total = []
Kappa1_pred_total = [] 
Kappa2_pred_total = [] 
Kappa3_pred_total = [] 
Kappa4_pred_total = [] 
Kappa5_pred_total = [] 


from datetime import datetime
now = datetime.now()
# dt_string = now.strftime("%m-%d-%H-%M")
dt_string = now.strftime("%m-%d")
dt_string = '03-07'


# for j in [1,2,3,4,5,6,7,8,9,10]:
for j in np.arange(1,11,1):

    casenumber ='set' +str(j)
    current_directory = os.getcwd()
    relative_path_results = '/Model5/Train-Results-'+dt_string+'-'+casenumber+'/'
    read_results_to = current_directory + relative_path_results 
    
    S_pred = np.loadtxt(read_results_to + 'S.txt') 
    I_pred = np.loadtxt(read_results_to + 'I.txt') 
    D_pred = np.loadtxt(read_results_to + 'D.txt')
    H_pred = np.loadtxt(read_results_to + 'H.txt')
    R_pred = np.loadtxt(read_results_to + 'R.txt')
    BetaI_pred = np.loadtxt(read_results_to + 'BetaI.txt')
    # Kappa_pred = np.loadtxt(read_results_to + 'Kappa.txt')
    Kappa1_pred = np.loadtxt(read_results_to + 'Kappa1.txt')
    Kappa2_pred = np.loadtxt(read_results_to + 'Kappa2.txt')
    Kappa3_pred = np.loadtxt(read_results_to + 'Kappa3.txt')
    Kappa4_pred = np.loadtxt(read_results_to + 'Kappa4.txt')
    Kappa5_pred = np.loadtxt(read_results_to + 'Kappa5.txt')
    p_pred = np.loadtxt(read_results_to + 'p.txt')
    q_pred = np.loadtxt(read_results_to + 'q.txt')
    I_new_pred = np.loadtxt(read_results_to + 'I_new.txt')
    H_new_pred = np.loadtxt(read_results_to + 'H_new.txt')
    D_new_pred = np.loadtxt(read_results_to + 'D_new.txt')
    I_sum_pred = np.loadtxt(read_results_to + 'I_sum.txt')
    H_sum_pred = np.loadtxt(read_results_to + 'H_sum.txt')
    D_sum_pred = np.loadtxt(read_results_to + 'D_sum.txt') 

    S_pred_total.append(S_pred) 
    I_pred_total.append(I_pred) 
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
    # Kappa_pred_total.append(Kappa_pred)
    Kappa1_pred_total.append(Kappa1_pred) 
    Kappa2_pred_total.append(Kappa2_pred) 
    Kappa3_pred_total.append(Kappa3_pred) 
    Kappa4_pred_total.append(Kappa4_pred) 
    Kappa5_pred_total.append(Kappa5_pred) 
    p_pred_total.append(p_pred)
    q_pred_total.append(q_pred) 
    
#%%
#Average  
S_pred_mean = np.mean(S_pred_total, axis=0) 
I_pred_mean = np.mean(I_pred_total, axis=0) 
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
# Kappa_pred_mean = np.mean(Kappa_pred_total, axis=0) 
Kappa1_pred_mean = np.mean(Kappa1_pred_total, axis=0)
Kappa2_pred_mean = np.mean(Kappa2_pred_total, axis=0)
Kappa3_pred_mean = np.mean(Kappa3_pred_total, axis=0)
Kappa4_pred_mean = np.mean(Kappa4_pred_total, axis=0)
Kappa5_pred_mean = np.mean(Kappa5_pred_total, axis=0)
p_pred_mean = np.mean(p_pred_total, axis=0) 
q_pred_mean = np.mean(q_pred_total, axis=0) 
 
S_pred_std = np.std(S_pred_total, axis=0) 
I_pred_std = np.std(I_pred_total, axis=0) 
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
# Kappa_pred_std = np.std(Kappa_pred_total, axis=0) 
Kappa1_pred_std = np.std(Kappa1_pred_total, axis=0) 
Kappa2_pred_std = np.std(Kappa2_pred_total, axis=0) 
Kappa3_pred_std = np.std(Kappa3_pred_total, axis=0)
Kappa4_pred_std = np.std(Kappa4_pred_total, axis=0) 
Kappa5_pred_std = np.std(Kappa5_pred_total, axis=0) 
p_pred_std = np.std(p_pred_total, axis=0) 
q_pred_std = np.std(q_pred_total, axis=0) 

#%%
#save results  
current_directory = os.getcwd()
relative_path_results = '/Model5/Train-Results-'+dt_string+'-Average/'
save_results_to = current_directory + relative_path_results 
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

np.savetxt(save_results_to + 'S_pred_mean.txt', S_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'I_pred_mean.txt', I_pred_mean.reshape((-1,1)))  
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
# np.savetxt(save_results_to + 'Kappa_pred_mean.txt', Kappa_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'Kappa1_pred_mean.txt', Kappa1_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa2_pred_mean.txt', Kappa2_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa3_pred_mean.txt', Kappa3_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa4_pred_mean.txt', Kappa4_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa5_pred_mean.txt', Kappa5_pred_mean.reshape((-1,1)))   
np.savetxt(save_results_to + 'p_pred_mean.txt', p_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'q_pred_mean.txt', q_pred_mean.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_pred_std.txt', S_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'I_pred_std.txt', I_pred_std.reshape((-1,1)))  
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
# np.savetxt(save_results_to + 'Kappa_pred_std.txt', Kappa_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'Kappa1_pred_std.txt', Kappa1_pred_std.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa2_pred_std.txt', Kappa2_pred_std.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa3_pred_std.txt', Kappa3_pred_std.reshape((-1,1)))
np.savetxt(save_results_to + 'Kappa4_pred_std.txt', Kappa4_pred_std.reshape((-1,1)))   
np.savetxt(save_results_to + 'Kappa5_pred_std.txt', Kappa5_pred_std.reshape((-1,1)))   
np.savetxt(save_results_to + 'p_pred_std.txt', p_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'q_pred_std.txt', q_pred_std.reshape((-1,1)))  


#%% 
first_date = t_data.iloc[6]
last_date = t_data.iloc[-1]

first_date = first_date[6:]+'-'+first_date[0:2]+'-'+first_date[3:5]
last_date = last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])+1)

date_total = np.arange(first_date, last_date, dtype='datetime64[D]')

# date_total = np.arange('2020-03-06', '2021-01-29', dtype='datetime64[D]')

Rc_pred_mean = BetaI_pred_mean/(1.0/6.0)
sf = 1e-6

#%%
# iteration = np.arange(0,100*Kappa_pred.shape[0],100)


#History of parameters
font = 32
# fig, ax = plt.subplots()
# plt.locator_params(axis='x', nbins=6)
# plt.locator_params(axis='y', nbins=6)
# # plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
# plt.xlabel('iteration', fontsize = font)
# plt.ylabel('model parameters', fontsize = font)
# # plt.yscale('log')
# plt.grid(True)  
# # ax.axhline(y=0.6, xmin=0.045, xmax=0.96, color='c', ls='--', lw=4)
# # ax.axhline(y=0.1, xmin=0.045, xmax=0.96, color='m', ls='--', lw=4)

# plt.plot(iteration, Kappa1_pred_mean,'b-', lw = 4, label='$\kappa_1$ (mean)') 
# ax.fill_between(iteration, (Kappa1_pred_mean-Kappa1_pred_std), (Kappa1_pred_mean+Kappa1_pred_std), facecolor = 'gray')#, label = '$\kappa_1$ (std)')

# plt.plot(iteration, Kappa2_pred_mean,'r--', lw = 4, label='$\kappa_2$ (mean)') 
# ax.fill_between(iteration, (Kappa2_pred_mean-Kappa2_pred_std), (Kappa2_pred_mean+Kappa2_pred_std), facecolor = 'gray')#, label = '$\kappa_2$ (std)')
 
# plt.plot(iteration, Kappa3_pred_mean,'g-.', lw = 4, label='$\kappa_3$ (mean)') 
# ax.fill_between(iteration, (Kappa3_pred_mean-Kappa3_pred_std), (Kappa3_pred_mean+Kappa3_pred_std), facecolor = 'gray')

# plt.plot(iteration, BetaI_pred_mean,'k:', lw = 5, label='$beta_I$ (fixed)') 
# # ax.fill_between(iteration, (BetaI_pred_mean-BetaI_pred_std), (BetaI_pred_mean+BetaI_pred_std), facecolor = 'gray', label = 'std')

# ax.set_ylim(-0.03,1.03)

# plt.legend(loc="lower left",  fontsize = 22, ncol=2)
# ax.tick_params(axis='both', labelsize = 24)
# fig.set_size_inches(w=12, h=7)
# plt.savefig(save_results_to + 'History_parameters.pdf', dpi=300)
# plt.savefig(save_results_to + 'History_parameters.png', dpi=300)




#%%
#Current Suspectious 
fig, ax = plt.subplots() 
ax.plot(date_total, S_SIHDR, 'ro', lw=5, markersize=10, label='SIHDR Model')
ax.plot(date_total, S_pred_mean/sf, 'k-', lw=5, label='fPINN') 
plt.fill_between(date_total.flatten(), \
                  (S_pred_mean+S_pred_std)/sf, \
                  (S_pred_mean-S_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$S$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Suspectious.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Suspectious.png', dpi=300) 

#%%
#Current Infectious 
fig, ax = plt.subplots() 
ax.plot(date_total, I_SIHDR, 'ro', lw=5, markersize=10, label='SIHDR Model')
ax.plot(date_total, I_pred_mean/sf, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (I_pred_mean+I_pred_std)/sf, \
                  (I_pred_mean-I_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$I$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Infectious.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Infectious.png', dpi=300) 

#%%
#Current Hos 
fig, ax = plt.subplots() 
ax.plot(date_total, H_SIHDR, 'ro', lw=5, markersize=10, label='SIHDR Model')
ax.plot(date_total, H_pred_mean/sf, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (H_pred_mean+H_pred_std)/sf, \
                  (H_pred_mean-H_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$H$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Hospitalized.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Hospitalized.png', dpi=300) 


#%%
#Current Death 
fig, ax = plt.subplots() 
ax.plot(date_total, D_SIHDR, 'ro', lw=5, markersize=10, label='SIHDR Model')
ax.plot(date_total, D_pred_mean/sf, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (D_pred_mean+D_pred_std)/sf, \
                  (D_pred_mean-D_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$D$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Death.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Death.png', dpi=300) 



#%%
#Current Removed 
fig, ax = plt.subplots() 
ax.plot(date_total, R_SIHDR, 'ro', lw=5, markersize=10, label='SIHDR Model')
ax.plot(date_total, R_pred_mean/sf, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (R_pred_mean+R_pred_std)/sf, \
                  (R_pred_mean-R_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$R$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Removed.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Removed.png', dpi=300) 


#%%
fig, ax = plt.subplots() 
ax.plot(date_total, I_new_train, 'ro', lw=5, markersize=5, label='Data-7davg')
ax.plot(date_total[1:], I_new_pred_mean/sf, 'k-', lw=3, label = 'fPINN (mean)')  
ax.fill_between(date_total[1:], (I_new_pred_mean-I_new_pred_std)/sf, (I_new_pred_mean+I_new_pred_std)/sf, facecolor = 'gray', label = 'fPINN (std)')
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=6)
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=18, ncol = 1, loc = 'upper center')
ax.tick_params(axis='x', labelsize = 18)
ax.tick_params(axis='y', labelsize = 18)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# plt.rc('font', size=12)
# ax.grid(True)
#ax.set_xlabel('Dates', fontsize = font-5) 
ax.set_ylabel('$I_{new}$', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=12, h=6.5)
plt.savefig(save_results_to + 'New_Infectious.pdf', dpi=150) 
plt.savefig(save_results_to + 'New_Infectious.png', dpi=150)


#%%
fig, ax = plt.subplots() 
ax.plot(date_total, H_new_train, 'ro', lw=5, markersize=5, label='Data-7davg')
ax.plot(date_total[1:], H_new_pred_mean/sf, 'k-', lw=3, label = 'fPINN (mean)')  
ax.fill_between(date_total[1:], (H_new_pred_mean-H_new_pred_std)/sf, (H_new_pred_mean+H_new_pred_std)/sf, facecolor = 'gray', label = 'fPINN (std)')
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=6)
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=18, ncol = 1, loc = 'upper center')
ax.tick_params(axis='x', labelsize = 18)
ax.tick_params(axis='y', labelsize = 18)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# plt.rc('font', size=12)
# ax.grid(True)
#ax.set_xlabel('Dates', fontsize = font-5) 
ax.set_ylabel('$H_{new}$', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=12, h=6.5)
plt.savefig(save_results_to + 'New_Hospitalized.pdf', dpi=150) 
plt.savefig(save_results_to + 'New_Hospitalized.png', dpi=150)

#%%
fig, ax = plt.subplots() 
ax.plot(date_total, D_new_train, 'ro', lw=5, markersize=5, label='Data-7davg')
ax.plot(date_total[1:], D_new_pred_mean/sf, 'k-', lw=3, label = 'fPINN (mean)')  
ax.fill_between(date_total[1:], (D_new_pred_mean-D_new_pred_std)/sf, 
                (D_new_pred_mean+D_new_pred_std)/sf, facecolor = 'gray', label = 'fPINN (std)')
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=6)
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=18, ncol = 1, loc = 'upper center')
ax.tick_params(axis='x', labelsize = 18)
ax.tick_params(axis='y', labelsize = 18)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# plt.rc('font', size=12)
# ax.grid(True)
#ax.set_xlabel('Dates', fontsize = font-5) 
ax.set_ylabel('$D_{new}$', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=12, h=6.5)
plt.savefig(save_results_to + 'New_Death.pdf', dpi=150) 
plt.savefig(save_results_to + 'New_Death.png', dpi=150)


#%%
#Kappa
fig, ax = plt.subplots() 
ax.plot(date_total, Kappa1_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa1_pred_mean+Kappa1_pred_std), \
                  (Kappa1_pred_mean-Kappa1_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_1$', fontsize = 80) 
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Kappa1.pdf', dpi=300) 
plt.savefig(save_results_to + 'Kappa1.png', dpi=300)  

#%%
#Kappa
fig, ax = plt.subplots() 
ax.plot(date_total, Kappa2_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa2_pred_mean+Kappa2_pred_std), \
                  (Kappa2_pred_mean-Kappa2_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_2$', fontsize = 80) 
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Kappa2.pdf', dpi=300) 
plt.savefig(save_results_to + 'Kappa2.png', dpi=300)
#%%
#Kappa
fig, ax = plt.subplots() 
ax.plot(date_total, Kappa3_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa3_pred_mean+Kappa3_pred_std), \
                  (Kappa3_pred_mean-Kappa3_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_3$', fontsize = 80) 
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Kappa3.pdf', dpi=300) 
plt.savefig(save_results_to + 'Kappa3.png', dpi=300)
#%%
#Kappa
fig, ax = plt.subplots() 
ax.plot(date_total, Kappa4_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa4_pred_mean+Kappa4_pred_std), \
                  (Kappa4_pred_mean-Kappa4_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_4$', fontsize = 80) 
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Kappa4.pdf', dpi=300) 
plt.savefig(save_results_to + 'Kappa4.png', dpi=300)
#%%
#Kappa
fig, ax = plt.subplots() 
ax.plot(date_total, Kappa5_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa5_pred_mean+Kappa5_pred_std), \
                  (Kappa5_pred_mean-Kappa5_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_5$', fontsize = 80) 
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Kappa5.pdf', dpi=300) 
plt.savefig(save_results_to + 'Kappa5.png', dpi=300)


#%%
# #New infectious
# fig, ax = plt.subplots() 
# ax.plot(date_total, I_new_train/sf, 'b-', lw=5, markersize=10, label='7-day averaged')
# # ax.set_xlim(0-0.5,180)
# # ax.set_ylim(0-0.5,6000+0.5)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
# plt.xticks(rotation=30)
# ax.legend(fontsize=40, ncol = 1, loc = 'best')
# ax.tick_params(axis='x', labelsize = 30)
# ax.tick_params(axis='y', labelsize = 30)
# # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# # plt.rc('font', size=20)
# # ax.grid(True)
# # ax.set_xlabel('Date', fontsize = font) 
# ax.set_ylabel('daily cases', fontsize = 60) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# fig.set_size_inches(w=25, h=12.5)
# plt.savefig(save_results_to + 'New_Infectious_raw.pdf', dpi=300) 
# plt.savefig(save_results_to + 'New_Infectious_raw.png', dpi=300) 


