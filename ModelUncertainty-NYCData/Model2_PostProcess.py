# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:17:46 2021

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
###Load data
data_frame = pandas.read_csv('Data/data-by-day.csv')  
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



t_star = np.arange(len(I_new_star))
t_star = t_star.reshape([len(t_star),1])
N = 8.399e6
I_new_train = I_new_star 
D_new_train = D_new_star 
H_new_train = H_new_star
I_sum_train = I_sum_star 
D_sum_train = D_sum_star 
H_sum_train = H_sum_star 


# data_frame = pandas.read_csv('Data/COVID-19 Rhode Island Data - Trends.csv')  
# date = data_frame['Date']
# newI = data_frame['New people who tested positive (counts first positive lab per person)'] #T x 1 array 
# newH = data_frame['New hospital admissions'] #T x 1 array 
# currentH = data_frame['Currently hospitalized'] #T x 1 array
# newD = data_frame['Hospital deaths'] #T x 1 array
# #Remove the nan values 
# date_nan = date.shape[0]-date.count()
# newI_nan = newI.shape[0]-newI.count()
# newH_nan = newH.shape[0]-newH.count()
# H_nan = currentH.shape[0]-currentH.count()
# newD_nan = newD.shape[0]-newD.count()
# Num_nan = max(np.array([date_nan, newI_nan, newH_nan, H_nan, newD_nan]))
# Date_total = date[0:-Num_nan]
# I_new = newI[0:-Num_nan]
# H_new = newH[0:-Num_nan]
# H = currentH[0:-Num_nan]
# D_new = newD[0:-Num_nan]  
# #7 days averaged data 
# I_new_star = I_new.rolling(window=7).mean()
# H_new_star = H_new.rolling(window=7).mean()
# H_star = H.rolling(window=7).mean()
# D_new_star = D_new.rolling(window=7).mean()

# I_new_star = I_new_star.to_numpy(dtype=np.float64)
# H_new_star = H_new_star.to_numpy(dtype=np.float64)
# H_star = H_star.to_numpy(dtype=np.float64)
# D_new_star = D_new_star.to_numpy(dtype=np.float64)

# Date_total = Date_total[6:]
# I_new_star = I_new_star[6:]
# H_new_star = H_new_star[6:]
# H_star = H_star[6:]
# D_new_star = D_new_star[6:] 
# I_new_star = I_new_star.reshape([len(I_new_star), 1])
# H_new_star = H_new_star.reshape([len(H_new_star), 1])
# H_star = H_star.reshape([len(H_star), 1])
# D_new_star = D_new_star.reshape([len(D_new_star), 1]) 

# I_sum_star = np.cumsum(I_new_star) 
# H_sum_star = np.cumsum(H_new_star)
# D_sum_star = np.cumsum(D_new_star)
# I_sum_star = I_sum_star.reshape([len(I_sum_star), 1]) 
# H_sum_star = H_sum_star.reshape([len(H_sum_star), 1]) 
# D_sum_star = D_sum_star.reshape([len(D_sum_star), 1])


#%%
#read results  

S_pred_total = []
E_pred_total = []
PreS_pred_total = []
Qua_pred_total = []
I_pred_total = []
J_pred_total = []
D_pred_total = []
H_pred_total = []
R_pred_total = [] 
BetaI_pred_total = []
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
dt_string = '03-04'


Model = '2'
relative_path_results0 = '/Model'+Model


#for j in [1,2,3,4,5]:
for j in np.arange(1,11,1):

    casenumber ='set' +str(j)
    current_directory = os.getcwd()
    relative_path_results = relative_path_results0 + '/Train-Results-'+dt_string+'-'+casenumber+'/'
    read_results_to = current_directory + relative_path_results 
    
    S_pred = np.loadtxt(read_results_to + 'S.txt')
    E_pred = np.loadtxt(read_results_to + 'E.txt')
    if Model == '3' or Model == '2':
        PreS_pred = np.loadtxt(read_results_to + 'PreS.txt')
    if Model == '3':
        Qua_pred = np.loadtxt(read_results_to + 'Qua.txt')
    I_pred = np.loadtxt(read_results_to + 'I.txt')
    J_pred = np.loadtxt(read_results_to + 'J.txt')
    D_pred = np.loadtxt(read_results_to + 'D.txt')
    H_pred = np.loadtxt(read_results_to + 'H.txt')
    R_pred = np.loadtxt(read_results_to + 'R.txt')
    BetaI_pred = np.loadtxt(read_results_to + 'BetaI.txt')
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
    if Model == '3' or Model == '2':
        PreS_pred_total.append(PreS_pred)
    if Model == '3':
        Qua_pred_total.append(Qua_pred)
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
    p_pred_total.append(p_pred)
    q_pred_total.append(q_pred) 
    
#%%
#Average  
S_pred_mean = np.mean(S_pred_total, axis=0)
E_pred_mean = np.mean(E_pred_total, axis=0)
if Model == '3' or Model == '2':
    PreS_pred_mean = np.mean(PreS_pred_total, axis=0)
if Model == '3':
    Qua_pred_mean = np.mean(Qua_pred_total, axis=0)
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
p_pred_mean = np.mean(p_pred_total, axis=0) 
q_pred_mean = np.mean(q_pred_total, axis=0) 

 
S_pred_std = np.std(S_pred_total, axis=0)
E_pred_std = np.std(E_pred_total, axis=0)
if Model == '3' or Model == '2':
    PreS_pred_std = np.std(PreS_pred_total, axis=0)
if Model == '3':
    Qua_pred_std = np.std(Qua_pred_total, axis=0)
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
p_pred_std = np.std(p_pred_total, axis=0) 
q_pred_std = np.std(q_pred_total, axis=0) 

#%%
#save results  
current_directory = os.getcwd()
relative_path_results = relative_path_results0 + '/Train-Results-'+dt_string+'-Average/'
save_results_to = current_directory + relative_path_results 
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

np.savetxt(save_results_to + 'S_pred_mean.txt', S_pred_mean.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_pred_mean.txt', E_pred_mean.reshape((-1,1)))  
if Model == '3' or Model == '2':
    np.savetxt(save_results_to + 'PreS_pred_mean.txt', PreS_pred_mean.reshape((-1,1))) 
if Model == '3':
    np.savetxt(save_results_to + 'Qua_pred_mean.txt', Qua_pred_mean.reshape((-1,1)))
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
np.savetxt(save_results_to + 'p_pred_mean.txt', p_pred_mean.reshape((-1,1)))  
np.savetxt(save_results_to + 'q_pred_mean.txt', q_pred_mean.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_pred_std.txt', S_pred_std.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_pred_std.txt', E_pred_std.reshape((-1,1))) 
if Model == '3' or Model == '2':
    np.savetxt(save_results_to + 'PreS_pred_std.txt', PreS_pred_std.reshape((-1,1))) 
if Model == '3':
    np.savetxt(save_results_to + 'Qua_pred_std.txt', Qua_pred_std.reshape((-1,1)))
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
np.savetxt(save_results_to + 'p_pred_std.txt', p_pred_std.reshape((-1,1)))  
np.savetxt(save_results_to + 'q_pred_std.txt', q_pred_std.reshape((-1,1)))  



#%% 
# first_date = Date_total.iloc[0]
# last_date = Date_total.iloc[-1]

# first_date = first_date[6:]+'-'+first_date[0:2]+'-'+first_date[3:5]
# last_date = last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])+1)

# date_total = np.arange(first_date, last_date, dtype='datetime64[D]')[:,None] 


first_date = '2020-03-06' #first_date[6:]+'-'+first_date[0:2]+'-'+first_date[3:5]
last_date = '2021-03-01' #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])+1)

date_total = np.arange(first_date, last_date, dtype='datetime64[D]')



# date_total = Date_total
sf = 1e-4
plt.rc('font', size=30)

#%%
#Current Suspectious 
fig, ax = plt.subplots() 
ax.plot(date_total, S_pred_mean/sf, 'k-', lw=5) 
plt.fill_between(date_total, \
                  (S_pred_mean+S_pred_std)/sf, \
                  (S_pred_mean-S_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=40, ncol = 1, loc = 'upper right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
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
#Current Exposed 
fig, ax = plt.subplots() 
ax.plot(date_total, E_pred_mean/sf, 'k-', lw=5) 
plt.fill_between(date_total, \
                  (E_pred_mean+E_pred_std)/sf, \
                  (E_pred_mean-E_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$E$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Exposed.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Exposed.png', dpi=300) 

#%%
#Current Presymp
if Model == '3' or Model == '2':
    fig, ax = plt.subplots() 
    ax.plot(date_total, PreS_pred_mean/sf, 'k-', lw=5) 
    plt.fill_between(date_total, \
                      (PreS_pred_mean+PreS_pred_std)/sf, \
                      (PreS_pred_mean-PreS_pred_std)/sf, \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
     
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=30)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$P$', fontsize = 80) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Current_preSymp.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Current_preSymp.png', dpi=300) 

#%%
#Current Quarantine 
if Model == '3':
    fig, ax = plt.subplots() 
    ax.plot(date_total, Qua_pred_mean/sf, 'k-', lw=5) 
    plt.fill_between(date_total, \
                      (Qua_pred_mean+Qua_pred_std)/sf, \
                      (Qua_pred_mean-Qua_pred_std)/sf, \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
     
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=30)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$Q$', fontsize = 80) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Current_Quarantine.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Current_Quarantine.png', dpi=300) 
    

#%%
#Current Asym
fig, ax = plt.subplots() 
ax.plot(date_total, J_pred_mean/sf, 'k-', lw=5) 
plt.fill_between(date_total, \
                  (J_pred_mean+J_pred_std)/sf, \
                  (J_pred_mean-J_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$J$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Assymp.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Assymp.png', dpi=300) 



#%%
#Current Infectious 
fig, ax = plt.subplots() 
ax.plot(date_total, I_pred_mean/sf, 'k-', lw=5)  
plt.fill_between(date_total, \
                  (I_pred_mean+I_pred_std)/sf, \
                  (I_pred_mean-I_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
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
#Current Hospitalized   
fig, ax = plt.subplots() 
ax.plot(date_total, H_pred_mean/sf, 'k-', lw=5, label = 'PINNs')  
plt.fill_between(date_total, \
                  (H_pred_mean+H_pred_std)/sf, \
                  (H_pred_mean-H_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$H$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Current_Hosppitalized.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Hosppitalized.png', dpi=300) 

#%%
#Current death   
fig, ax = plt.subplots() 
ax.plot(date_total, D_pred_mean/sf, 'k-', lw=5)  
plt.fill_between(date_total, \
                  (D_pred_mean+D_pred_std)/sf, \
                  (D_pred_mean-D_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
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
ax.plot(date_total, R_pred_mean/sf, 'k-', lw=5)  
plt.fill_between(date_total, \
                  (R_pred_mean+R_pred_std)/sf, \
                  (R_pred_mean-R_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
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
#New infectious
fig, ax = plt.subplots() 
ax.plot(date_total[1:], I_new_pred_mean/sf, 'k-', lw=5, label = 'PINNs')  
ax.plot(date_total, I_new_star, 'ro', markersize=8, label='Data-7davg')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{new}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'New_Infectious.pdf', dpi=300) 
plt.savefig(save_results_to + 'New_Infectious.png', dpi=300) 

#%%
#Cumulative Infectious 
fig, ax = plt.subplots() 
ax.plot(date_total, I_sum_star, 'ro', markersize=8, label='Data-7davg')
ax.plot(date_total, I_sum_pred_mean/sf, 'k-', lw=5, label = 'PINNs')  
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$I^{c}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Cumulative_Infectious.pdf', dpi=300) 
plt.savefig(save_results_to + 'Cumulative_Infectious.png', dpi=300) 

#%%
#New Hosptalized
fig, ax = plt.subplots() 
ax.plot(date_total[1:], H_new_pred_mean/sf, 'k-', lw=5, label = 'PINNs')  
ax.plot(date_total, H_new_star, 'ro', markersize=8, label='Data-7davg')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$H^{new}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'New_Hosptalized.pdf', dpi=300) 
plt.savefig(save_results_to + 'New_Hosptalized.png', dpi=300) 

#%%
#Cumulative Hosptalized 
fig, ax = plt.subplots() 
ax.plot(date_total, H_sum_star, 'ro', markersize=8, label='Data-7davg')
ax.plot(date_total, H_sum_pred_mean/sf, 'k-', lw=5, label = 'PINNs')  
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$H^{c}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Cumulative_Hosptalized.pdf', dpi=300) 
plt.savefig(save_results_to + 'Cumulative_Hosptalized.png', dpi=300) 


#%%
#New death
fig, ax = plt.subplots() 
ax.plot(date_total[1:], D_new_pred_mean/sf, 'k-', lw=5, label = 'PINNs')  
ax.plot(date_total, D_new_star, 'ro', markersize=8, label='Data-7davg')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=40, ncol = 1, loc = 'upper left')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$D^{new}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'New_Death.pdf', dpi=300) 
plt.savefig(save_results_to + 'New_Death.png', dpi=300) 

#%%
#Cumulative Death 
fig, ax = plt.subplots() 
ax.plot(date_total, D_sum_pred_mean/sf, 'k-', lw=5, label = 'PINNs')  
ax.plot(date_total, D_sum_star, 'ro', markersize=8, label='Data-7davg')
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$D^{c}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'Cumulative_Death.pdf', dpi=300) 
plt.savefig(save_results_to + 'Cumulative_Death.png', dpi=300) 

#%%
#BetaI
fig, ax = plt.subplots() 
ax.plot(date_total, BetaI_pred_mean, 'k-', lw=5)  
plt.fill_between(date_total, \
                  BetaI_pred_mean+BetaI_pred_std, \
                  BetaI_pred_mean-BetaI_pred_std, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$Beta_{I}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'BetaI.pdf', dpi=300) 
plt.savefig(save_results_to + 'BetaI.png', dpi=300)  


#%%
#p
fig, ax = plt.subplots() 
ax.plot(date_total, p_pred_mean, 'k-', lw=5)  
plt.fill_between(date_total, \
                  p_pred_mean+p_pred_std, \
                  p_pred_mean-p_pred_std, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$p$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'p.pdf', dpi=300) 
plt.savefig(save_results_to + 'p.png', dpi=300)  

#%%
#q
fig, ax = plt.subplots() 
ax.plot(date_total, q_pred_mean, 'k-', lw=5, label='PINNs')  
plt.fill_between(date_total, \
                  q_pred_mean+q_pred_std, \
                  q_pred_mean-q_pred_std, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) 
# ax.set_xlim(0-0.5,180)
ax.set_ylim(-0.1,1.1)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=18, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('$q$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to + 'q.pdf', dpi=300) 
plt.savefig(save_results_to + 'q.png', dpi=300)  






