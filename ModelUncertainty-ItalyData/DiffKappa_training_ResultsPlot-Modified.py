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

#Load data 
I_star = np.loadtxt('Data/Infectious.txt') #T x 1 array 
I_star = I_star.reshape([len(I_star),1])
R_star = np.loadtxt('Data/Recovered.txt') #T x 1 array
R_star = R_star.reshape([len(R_star),1])
D_star = np.loadtxt('Data/Death.txt') #T x 1 array  
D_star = D_star.reshape([len(D_star),1])
t_star = np.arange(len(I_star))
t_star = t_star[:,None]
N = 60461826 

S_star = N - I_star - R_star - D_star
S_star = S_star.reshape([len(S_star),1])

#%%
#load results   

from datetime import datetime
now = datetime.now()
# dt_string = now.strftime("%m-%d-%H-%M")
# dt_string = now.strftime("%m-%d")
dt_string = '03-19' 

current_directory = os.getcwd()
relative_path_results = '/SIRD-DiffKappa-Beta/Train-Results-'+dt_string+'-Average/'
load_results_to = current_directory + relative_path_results  
    

current_directory = os.getcwd()
relative_path_results = '/SIRD-DiffKappa-Beta/Results-Plot-'+dt_string+'/'
save_results_to = current_directory + relative_path_results 
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    

S_pred_mean = np.loadtxt(load_results_to + 'S_pred_mean.txt')   
I_pred_mean = np.loadtxt(load_results_to + 'I_pred_mean.txt')  
D_pred_mean = np.loadtxt(load_results_to + 'D_pred_mean.txt')  
R_pred_mean = np.loadtxt(load_results_to + 'R_pred_mean.txt')  
Kappa1_pred_mean = np.loadtxt(load_results_to + 'Kappa1_pred_mean.txt')   
Kappa2_pred_mean = np.loadtxt(load_results_to + 'Kappa2_pred_mean.txt')   
Kappa3_pred_mean = np.loadtxt(load_results_to + 'Kappa3_pred_mean.txt')   
Kappa4_pred_mean = np.loadtxt(load_results_to + 'Kappa4_pred_mean.txt')   
Beta_pred_mean = np.loadtxt(load_results_to + 'Beta_pred_mean.txt')  

S_pred_std = np.loadtxt(load_results_to + 'S_pred_std.txt')  
I_pred_std = np.loadtxt(load_results_to + 'I_pred_std.txt')  
D_pred_std = np.loadtxt(load_results_to + 'D_pred_std.txt')  
R_pred_std = np.loadtxt(load_results_to + 'R_pred_std.txt')    
Kappa1_pred_std = np.loadtxt(load_results_to + 'Kappa1_pred_std.txt')   
Kappa2_pred_std = np.loadtxt(load_results_to + 'Kappa2_pred_std.txt')   
Kappa3_pred_std = np.loadtxt(load_results_to + 'Kappa3_pred_std.txt')
Kappa4_pred_std = np.loadtxt(load_results_to + 'Kappa4_pred_std.txt')  
Beta_pred_std = np.loadtxt(load_results_to + 'Beta_pred_std.txt') 


#%% 
date_total = np.arange('2020-01-22', '2021-02-28', dtype='datetime64[D]')
 
sf = 1e-7 
plt.rc('font', size=30)

#%%
#Current Suspectious 
fig, ax = plt.subplots() 
ax.plot(date_total, S_star, 'ro', lw=5, markersize=10, label='Data')
ax.plot(date_total, S_pred_mean/sf, 'k-', lw=5, label='fPINN fitting') 
plt.fill_between(date_total.flatten(), \
                  (S_pred_mean+S_pred_std)/sf, \
                  (S_pred_mean-S_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label = 'std') 


# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
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
#fig.set_size_inches(w=12, h=6)

ax2 = ax.twinx() 
ax2.set_ylim(0,1)
ax2.tick_params(axis='y', labelsize = 50)
color = 'tab:blue'
ax2.set_ylabel('$\kappa_1$', fontsize = 80, color=color) 

Kappa1 = np.where(t_star.flatten()<=27, 0.950, 0.900)
ax2.plot(date_total, Kappa1, color=color, ls='--', lw=5, label='piece-wise')  

ax2.plot(date_total, Kappa1_pred_mean, color=color, lw=5, label='fPINN inference')
plt.fill_between(date_total.flatten(), \
                  (Kappa1_pred_mean+Kappa1_pred_std), \
                  (Kappa1_pred_mean-Kappa1_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label='std') 
ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=7))    
ax2.legend(fontsize=40, ncol = 1, loc = 'lower center')


plt.savefig(save_results_to + 'Current_Suspectious.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Suspectious.png', dpi=300) 

#%%
#Current Infectious 
fig, ax = plt.subplots() 
ax.plot(date_total, I_star, 'ro', lw=5, markersize=10, label='Data')
ax.plot(date_total, I_pred_mean/sf, 'k-', lw=5, label='fPINN fitting')  
plt.fill_between(date_total.flatten(), \
                  (I_pred_mean+I_pred_std)/sf, \
                  (I_pred_mean-I_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label = 'std') 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'lower left')
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

ax2 = ax.twinx() 
ax2.set_ylim(0,1)
ax2.tick_params(axis='y', labelsize = 50)
color = 'tab:blue'
ax2.set_ylabel('$\kappa_2$', fontsize = 80, color=color) 

Kappa2 = np.where(t_star.flatten()<=27, 0.965, 0.913)
ax2.plot(date_total, Kappa2, color=color, ls='--', lw=5, label='piece-wise')      

ax2.plot(date_total, Kappa2_pred_mean, color=color, lw=5, label='fPINN inference')
plt.fill_between(date_total.flatten(), \
                  (Kappa2_pred_mean+Kappa2_pred_std), \
                  (Kappa2_pred_mean-Kappa2_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)
ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
ax2.legend(fontsize=40, ncol = 1, loc = 'lower center')

plt.savefig(save_results_to + 'Current_Infectious.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Infectious.png', dpi=300) 

#%%
#Current Death 
fig, ax = plt.subplots() 
ax.plot(date_total, D_star, 'ro', lw=5, markersize=10, label='Data')
ax.plot(date_total, D_pred_mean/sf, 'k-', lw=5, label='fPINN fitting')  
plt.fill_between(date_total.flatten(), \
                  (D_pred_mean+D_pred_std)/sf, \
                  (D_pred_mean-D_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label = 'std') 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'lower center')
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


ax2 = ax.twinx() 
ax2.set_ylim(0,1.02)
ax2.tick_params(axis='y', labelsize = 50)
color = 'tab:blue'
ax2.set_ylabel('$\kappa_3$', fontsize = 80, color=color) 

Kappa4 = np.where(t_star.flatten()<=27, 1.0, 0.987)
ax2.plot(date_total, Kappa4, color=color, ls='--', lw=5, label='piece-wise')  

ax2.plot(date_total, Kappa4_pred_mean, color=color, lw=5, label='fPINN inference')
plt.fill_between(date_total.flatten(), \
                  (Kappa4_pred_mean+Kappa4_pred_std), \
                  (Kappa4_pred_mean-Kappa4_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label='std') 

ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=7))    
ax2.legend(fontsize=40, ncol = 1, loc = 'lower right')

plt.savefig(save_results_to + 'Current_Death.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Death.png', dpi=300) 



#%%
#Current Removed 
fig, ax = plt.subplots() 
ax.plot(date_total, R_star, 'ro', lw=5, markersize=10, label='Data')
ax.plot(date_total, R_pred_mean/sf, 'k-', lw=5, label='fPINN fitting')  
plt.fill_between(date_total.flatten(), \
                  (R_pred_mean+R_pred_std)/sf, \
                  (R_pred_mean-R_pred_std)/sf, \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label = 'std') 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'lower center')
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

ax2 = ax.twinx() 
ax2.set_ylim(0,1.02)
ax2.tick_params(axis='y', labelsize = 50)
color = 'tab:blue'
ax2.set_ylabel('$\kappa_4$', fontsize = 80, color=color) 

Kappa3 = np.where(t_star.flatten()<=27, 1.0, 0.907)
ax2.plot(date_total, Kappa3, color=color, ls='--', lw=5, label='piece-wise')  

ax2.plot(date_total, Kappa3_pred_mean, color=color, lw=5, label='fPINN inference')
plt.fill_between(date_total.flatten(), \
                  (Kappa3_pred_mean+Kappa3_pred_std), \
                  (Kappa3_pred_mean-Kappa3_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label='std') 
ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=7))    
ax2.legend(fontsize=40, ncol = 1, loc = 'lower right')


plt.savefig(save_results_to + 'Current_Removed.pdf', dpi=300) 
plt.savefig(save_results_to + 'Current_Removed.png', dpi=300) 


#%%
#Kappa

Kappa1 = np.where(t_star.flatten()<=27, 0.950, 0.900)

fig, ax = plt.subplots() 
ax.plot(date_total, Kappa1_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa1_pred_mean+Kappa1_pred_std), \
                  (Kappa1_pred_mean-Kappa1_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label='std') 

ax.plot(date_total, Kappa1, 'm-', lw=5, label='Piecewise order')      
    
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_1$', fontsize = 80) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
#plt.savefig(save_results_to + 'Kappa1.pdf', dpi=300) 
#plt.savefig(save_results_to + 'Kappa1.png', dpi=300)  

#%%
#Kappa
Kappa2 = np.where(t_star.flatten()<=27, 0.965, 0.913)

fig, ax = plt.subplots() 
ax.plot(date_total, Kappa2_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa2_pred_mean+Kappa2_pred_std), \
                  (Kappa2_pred_mean-Kappa2_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label = 'std')

ax.plot(date_total, Kappa2, 'm-', lw=5, label='Piecewise order')      
     
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_2$', fontsize = 80) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
#plt.savefig(save_results_to + 'Kappa2.pdf', dpi=300) 
#plt.savefig(save_results_to + 'Kappa2.png', dpi=300)

#%%
#Kappa
Kappa3 = np.where(t_star.flatten()<=27, 1.0, 0.907)

fig, ax = plt.subplots() 
ax.plot(date_total, Kappa3_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa3_pred_mean+Kappa3_pred_std), \
                  (Kappa3_pred_mean-Kappa3_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label = 'std')

ax.plot(date_total, Kappa3, 'm-', lw=5, label='Piecewise order')      
 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_3$', fontsize = 80) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
#plt.savefig(save_results_to + 'Kappa3.pdf', dpi=300) 
#plt.savefig(save_results_to + 'Kappa3.png', dpi=300)

#%%
#Kappa
Kappa4 = np.where(t_star.flatten()<=27, 1.0, 0.987)

fig, ax = plt.subplots() 
ax.plot(date_total, Kappa4_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Kappa4_pred_mean+Kappa4_pred_std), \
                  (Kappa4_pred_mean-Kappa4_pred_std), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True)#, label = 'std') 

ax.plot(date_total, Kappa4, 'm-', lw=5, label='Piecewise order')      
 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$\kappa_4$', fontsize = 80) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
#plt.savefig(save_results_to + 'Kappa4.pdf', dpi=300) 
#plt.savefig(save_results_to + 'Kappa4.png', dpi=300)

#%%
#Beta

r0 = 1.2e-6
rt = np.where(t_star.flatten()<=27, r0, 0.66*r0*np.exp(-(t_star.flatten()-27)/2)+0.34*r0)
N = 60461826

fig, ax = plt.subplots() 
ax.plot(date_total, Beta_pred_mean, 'k-', lw=5, label='fPINN')  
plt.fill_between(date_total.flatten(), \
                  (Beta_pred_mean+Beta_pred_std*0.2), \
                  (Beta_pred_mean-Beta_pred_std*0.2), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) #, label = 'std') 
# ax.plot(date_total, rt, 'b-', lw=5, label='$r(t)\ in\ reference$')   
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$r_{(t)}$', fontsize = 80) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
#plt.savefig(save_results_to + 'Beta.pdf', dpi=300) 
#plt.savefig(save_results_to + 'Beta.png', dpi=300)

#%%
#Rc
a = 0.48e-2

fig, ax = plt.subplots() 
ax.plot(date_total, Beta_pred_mean/a, 'k-', lw=5, label='fPINN')  
# plt.fill_between(date_total.flatten(), \
#                   (Beta_pred_mean+Beta_pred_std), \
#                   (Beta_pred_mean-Beta_pred_std), \
#                   facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label = 'std') 
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.rc('font', size=30)
# ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$R_{c}$', fontsize = 80) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
#plt.savefig(save_results_to + 'Rc.pdf', dpi=300) 
#plt.savefig(save_results_to + 'Rc.png', dpi=300)

#%%
#r(t) 
r0 = 1.2e-6 
rt = np.where(t_star.flatten()<=27, r0, 0.66*r0*np.exp(-(t_star.flatten()-27)/2)+0.34*r0)
N = 10e5#60461826

fig, ax = plt.subplots() 
ax.plot(date_total, rt*N, 'm-', lw=5)    
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(-6,-6))
plt.rc('font', size=30)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('Piecewise $r_{(t)}$', fontsize = 50) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)
#plt.savefig(save_results_to + 'rt.pdf', dpi=300) 
#plt.savefig(save_results_to + 'rt.png', dpi=300)       
    
#%%
#Beta

r0 = 1.2e-6
rt = np.where(t_star.flatten()<=27, r0, 0.66*r0*np.exp(-(t_star.flatten()-27)/2)+0.34*r0)
N = 2.66e5
color = 'tab:blue'

fig, ax = plt.subplots() 
ax.plot(date_total, Beta_pred_mean, color=color, lw=5, label='fPINN inference')  
plt.fill_between(date_total.flatten(), \
                  (Beta_pred_mean+Beta_pred_std*0.2), \
                  (Beta_pred_mean-Beta_pred_std*0.2), \
                  facecolor=(0.1,0.2,0.5,0.3), interpolate=True) #, label = 'std') 

#ax.plot(date_total, rt*N, 'm-', lw=5, label='Piecewise $r(t)$')   
# ax.set_xlim(0-0.5,180)
# ax.set_ylim(0-0.5,6000+0.5)
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30) 
ax.legend(fontsize=40, ncol = 1, loc = 'upper center')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 50)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
plt.rc('font', size=30)
#ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel('$r(t) * N$', fontsize = 80) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
fig.set_size_inches(w=25, h=12.5)


ax2 = ax.twinx() 
#ax2.set_ylim(0,1.02)
ax2.tick_params(axis='y', labelsize = 50)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
color = 'tab:blue'
ax2.set_ylabel('$r(t) * N$', fontsize = 80, color=color) 

ax2.plot(date_total, rt*N, color=color, ls='--', lw=5, label='piece-wise')  

ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=7))    
ax2.legend(fontsize=40, ncol = 1, loc = 'upper right')




plt.savefig(save_results_to + 'RtCompare.pdf', dpi=300) 
plt.savefig(save_results_to + 'RtCompare.png', dpi=300)
