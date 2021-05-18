# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:46:24 2021

@author: Administrator 


"""


import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas
import math 
from math import gamma
from scipy.integrate import odeint
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


from datetime import datetime
now = datetime.now()
# dt_string = now.strftime("%m-%d-%H-%M")
# dt_string = now.strftime("%m-%d")
dt_string = '05-04'

# #Load data
# data_frame = pandas.read_csv('Data/michigan-history.csv')  
# newI = data_frame['positiveIncrease'] #T x 1 array 
# currentH = data_frame['hospitalizedCurrently'] #T x 1 array
# newD = data_frame['deathIncrease'] #T x 1 array
# #Remove the nan values 
# newI_nan = newI.shape[0]-newI.count()
# H_nan = currentH.shape[0]-currentH.count()
# newD_nan = newD.shape[0]-newD.count()
# Num_nan = max(np.array([newI_nan, H_nan, newD_nan]))
# I_new = newI[0:-Num_nan]
# H = currentH[0:-Num_nan]
# D_new = newD[0:-Num_nan] 
# #Reverse the series
# I_new = I_new.iloc[::-1] 
# H = H[::-1] 
# D_new = D_new[::-1]  
# #7 days averaged data 
# I_new_star = I_new.rolling(window=7).mean()
# H_star = H.rolling(window=7).mean()
# D_new_star = D_new.rolling(window=7).mean()

# I_new_star = I_new_star.to_numpy(dtype=np.float64)
# H_star = H_star.to_numpy(dtype=np.float64)
# D_new_star = D_new_star.to_numpy(dtype=np.float64)

# I_new_star = I_new_star[6:]
# H_star = H_star[6:]
# D_new_star = D_new_star[6:] 
# I_new_star = I_new_star.reshape([len(I_new_star), 1])
# H_star = H_star.reshape([len(H_star), 1])
# D_new_star = D_new_star.reshape([len(D_new_star), 1]) 

# I_sum_star = np.cumsum(I_new_star) 
# D_sum_star = np.cumsum(D_new_star)
# I_sum_star = I_sum_star.reshape([len(I_sum_star), 1]) 
# D_sum_star = D_sum_star.reshape([len(D_sum_star), 1])

# t_star = np.arange(len(I_new_star))
# t_star = t_star.reshape([len(t_star),1]) 

# N = 9992430 

# first_date = '2020-04-15' #first_date[6:]+'-'+first_date[0:2]+'-'+first_date[3:5]
# last_date = '2021-03-08' #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])+1)
# first_date_pred = '2021-03-07' #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])-1)
# last_date_pred = '2022-01-01'

#Load data
data_frame = pandas.read_csv('Data/MI_data.csv')
I_new = data_frame['new_cases'] #T x 1 array 
H = data_frame['inpatient_beds_used_covid'] #T x 1 array
D_new = data_frame['new_death']
#7 days averaged data 
I_new_star = I_new.rolling(window=7).mean()
H_star = H.rolling(window=7).mean()
D_new_star = D_new.rolling(window=7).mean()

I_new_star = I_new_star.to_numpy(dtype=np.float64)
H_star = H_star.to_numpy(dtype=np.float64)
D_new_star = D_new_star.to_numpy(dtype=np.float64)

I_new_star = I_new_star[6:]
H_star = H_star[6:]
D_new_star = D_new_star[6:] 
I_new_star = I_new_star.reshape([len(I_new_star), 1])
H_star = H_star.reshape([len(H_star), 1])
D_new_star = D_new_star.reshape([len(D_new_star), 1]) 

I_sum_star = np.cumsum(I_new_star) 
D_sum_star = np.cumsum(D_new_star)
I_sum_star = I_sum_star.reshape([len(I_sum_star), 1]) 
D_sum_star = D_sum_star.reshape([len(D_sum_star), 1])

t_star = np.arange(len(I_new_star))
t_star = t_star.reshape([len(t_star),1]) 

N = 9992430

first_date = '2020-03-16' #first_date[6:]+'-'+first_date[0:2]+'-'+first_date[3:5]
last_date = '2021-05-02' #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])+1)
first_date_pred = '2021-05-01' #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])-1)
last_date_pred = '2021-12-31'

date_total = np.arange(first_date, last_date, dtype='datetime64[D]')[:,None] 
data_mean = np.arange(first_date, last_date, dtype='datetime64[D]')[:,None]
data_pred = np.arange(first_date_pred, last_date_pred, dtype='datetime64[D]')[:,None] 


sf = 1e-4


# load data
BetaI_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/BetaI_pred_mean.txt') 
p_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/p_pred_mean.txt')
q_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/q_pred_mean.txt') 
t_mean = np.arange(len(BetaI_PINN))

S_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/S_pred_mean.txt') 
S_PINN = S_PINN/sf
E_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/E_pred_mean.txt') 
E_PINN = E_PINN/sf
PreS_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/PreS_pred_mean.txt') 
PreS_PINN = PreS_PINN/sf
I_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/I_pred_mean.txt') 
I_PINN = I_PINN/sf
J_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/J_pred_mean.txt') 
J_PINN = J_PINN/sf
H_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/H_pred_mean.txt') 
H_PINN = H_PINN/sf 
D_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/D_pred_mean.txt') 
D_PINN = D_PINN/sf
R_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/R_pred_mean.txt') 
R_PINN = R_PINN/sf
I_sum_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/I_sum_pred_mean.txt') 
I_sum_PINN = I_sum_PINN/sf
H_sum_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/H_sum_pred_mean.txt') 
H_sum_PINN = H_sum_PINN/sf
I_new_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/I_new_pred_mean.txt') 
I_new_PINN = I_new_PINN/sf
H_new_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/H_new_pred_mean.txt') 
H_new_PINN = H_new_PINN/sf
D_new_PINN = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/D_new_pred_mean.txt') 
D_new_PINN = D_new_PINN/sf 

#%%
# #Interpolations  
Beta_interp = scipy.interpolate.interp1d(t_mean.flatten(), BetaI_PINN.flatten(), fill_value="extrapolate")

#%%
    ######################################################################
    ################ Predicting by sloving forward problem ###############
    ######################################################################  
#%%    
#Initial conditions for ODE system  
S_init = float(S_PINN[-2]) 
E_init = float(E_PINN[-2]) 
PreS_init = float(PreS_PINN[-2])
I_init = float(I_PINN[-2]) 
J_init = float(J_PINN[-2])  
D_init = float(D_PINN[-2]) 
H_init = float(H_PINN[-2]) 
R_init = float(R_PINN[-2]) 
I_sum_init = float(I_sum_PINN[-2]) 
D_sum_init = float(D_PINN[-2]) 
H_sum_init = float(H_sum_PINN[-2])  
U_init = [S_init, E_init, PreS_init, I_init, J_init, D_init, H_init, R_init, I_sum_init, H_sum_init]

#Parameters     
Chi = 0.55
eps1 = 0.75
eps2 = 0.0
delta = 0.6
alpha = 1.0/5.2
alpha1 = 1.0/2.9
alpha2 = 1.0/2.3
gamma = 1.0/6.0
gammaA = 1.0/6.0
phiD = 1.0/15.0
phiR = 1.0/7.5 
p_mean = p_PINN[-2] 
q_mean = q_PINN[-2]
t_pred = np.arange(len(t_mean)-2, len(t_mean)+len(data_pred)-2)
t_pred = t_pred.reshape([len(t_pred),1])   

#%%
#Piecewise linear vacination 
def V_1(Dt):
    # return 0
    if Dt <= 260:
        return 0 
    else:
        return (Dt-260)*500   
def V_2(Dt):
    # return 0
    if Dt <= 260:
        return 0 
    else:
        return (Dt-260)*1000  

#ODEs  
def ODEs_mean_1(X, t, xi, Pert): 
    S, E, PreS, I, J, D, H, R, sumI, sumH = X  
    dSdt = -(BetaI_PINN[-1] * (1+xi*Pert) * (Chi * PreS + I + eps1 * J)/N) * S - V_1(t)/N*S 
    dEdt = (BetaI_PINN[-1] * (1+xi*Pert) * (Chi * PreS + I + eps1 * J)/N) * S - alpha1 * E 
    dPreSdt = alpha1* E - alpha2 * PreS
    dIdt = (delta*alpha2) * PreS - gamma * I 
    dJdt = ((1-delta)*alpha2) * PreS - gammaA * J 
    dDdt = (q_mean*phiD)*H
    dHdt = (p_mean*gamma)*I - (q_mean*phiD) * H - ((1-q_mean)*phiR) * H 
    dRdt = gammaA*J + ((1-p_mean)*gamma)*I + ((1-q_mean)*phiR)*H + V_1(t)/N*S 
    dsumIdt = (delta*alpha2)*PreS
    dsumHdt = (p_mean*gamma)*I 
    return [dSdt, dEdt, dPreSdt, dIdt, dJdt, dDdt, dHdt, dRdt, dsumIdt, dsumHdt] 

def ODEs_mean_2(X, t, xi, Pert): 
    S, E, PreS, I, J, D, H, R, sumI, sumH = X  
    dSdt = -(BetaI_PINN[-1] * (1+xi*Pert) * (Chi * PreS + I + eps1 * J)/N) * S - V_2(t)/N*S 
    dEdt = (BetaI_PINN[-1] * (1+xi*Pert) * (Chi * PreS + I + eps1 * J)/N) * S - alpha1 * E 
    dPreSdt = alpha1* E - alpha2 * PreS
    dIdt = (delta*alpha2) * PreS - gamma * I 
    dJdt = ((1-delta)*alpha2) * PreS - gammaA * J 
    dDdt = (q_mean*phiD)*H
    dHdt = (p_mean*gamma)*I - (q_mean*phiD) * H - ((1-q_mean)*phiR) * H 
    dRdt = gammaA*J + ((1-p_mean)*gamma)*I + ((1-q_mean)*phiR)*H + V_2(t)/N*S 
    dsumIdt = (delta*alpha2)*PreS
    dsumHdt = (p_mean*gamma)*I 
    return [dSdt, dEdt, dPreSdt, dIdt, dJdt, dDdt, dHdt, dRdt, dsumIdt, dsumHdt]

#%%
Pert0 = 0.10
Sol_ub_d0_V1 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (1,Pert0))  
S_ub_d0_V1 = Sol_ub_d0_V1[:,0]   
E_ub_d0_V1 = Sol_ub_d0_V1[:,1]   
PreS_ub_d0_V1 = Sol_ub_d0_V1[:,2]   
I_ub_d0_V1 = Sol_ub_d0_V1[:,3]   
J_ub_d0_V1 = Sol_ub_d0_V1[:,4]   
D_ub_d0_V1 = Sol_ub_d0_V1[:,5]   
H_ub_d0_V1 = Sol_ub_d0_V1[:,6]   
R_ub_d0_V1 = Sol_ub_d0_V1[:,7]   
sumI_ub_d0_V1 = Sol_ub_d0_V1[:,8] 
sumH_ub_d0_V1 = Sol_ub_d0_V1[:,9]  
newI_ub_d0_V1 = np.diff(sumI_ub_d0_V1)
newH_ub_d0_V1 = np.diff(sumH_ub_d0_V1)
newD_ub_d0_V1 = np.diff(D_ub_d0_V1)  
newI_ub_d0_V1 = newI_ub_d0_V1*I_new_PINN[-1]/newI_ub_d0_V1[0]
newH_ub_d0_V1 = newH_ub_d0_V1*H_new_PINN[-1]/newH_ub_d0_V1[0]
newD_ub_d0_V1 = newD_ub_d0_V1*D_new_PINN[-1]/newD_ub_d0_V1[0] 

Sol_lb_d0_V1 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (-1,Pert0))  
S_lb_d0_V1 = Sol_lb_d0_V1[:,0]   
E_lb_d0_V1 = Sol_lb_d0_V1[:,1]    
PreS_lb_d0_V1 = Sol_lb_d0_V1[:,2]  
I_lb_d0_V1 = Sol_lb_d0_V1[:,3]   
J_lb_d0_V1 = Sol_lb_d0_V1[:,4]   
D_lb_d0_V1 = Sol_lb_d0_V1[:,5]   
H_lb_d0_V1 = Sol_lb_d0_V1[:,6]   
R_lb_d0_V1 = Sol_lb_d0_V1[:,7]   
sumI_lb_d0_V1 = Sol_lb_d0_V1[:,8] 
sumH_lb_d0_V1 = Sol_lb_d0_V1[:,9]  
newI_lb_d0_V1 = np.diff(sumI_lb_d0_V1)
newH_lb_d0_V1 = np.diff(sumH_lb_d0_V1)
newD_lb_d0_V1 = np.diff(D_lb_d0_V1)  
newI_lb_d0_V1 = newI_lb_d0_V1*I_new_PINN[-1]/newI_lb_d0_V1[0]
newH_lb_d0_V1 = newH_lb_d0_V1*H_new_PINN[-1]/newH_lb_d0_V1[0]
newD_lb_d0_V1 = newD_lb_d0_V1*D_new_PINN[-1]/newD_lb_d0_V1[0] 


Sol_ub_d0_V2 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (1,Pert0))  
S_ub_d0_V2 = Sol_ub_d0_V2[:,0]   
E_ub_d0_V2 = Sol_ub_d0_V2[:,1]    
PreS_ub_d0_V2 = Sol_ub_d0_V2[:,2]   
I_ub_d0_V2 = Sol_ub_d0_V2[:,3]   
J_ub_d0_V2 = Sol_ub_d0_V2[:,4]   
D_ub_d0_V2 = Sol_ub_d0_V2[:,5]   
H_ub_d0_V2 = Sol_ub_d0_V2[:,6]   
R_ub_d0_V2 = Sol_ub_d0_V2[:,7]   
sumI_ub_d0_V2 = Sol_ub_d0_V2[:,8] 
sumH_ub_d0_V2 = Sol_ub_d0_V2[:,9]  
newI_ub_d0_V2 = np.diff(sumI_ub_d0_V2)
newH_ub_d0_V2 = np.diff(sumH_ub_d0_V2)
newD_ub_d0_V2 = np.diff(D_ub_d0_V2)  
newI_ub_d0_V2 = newI_ub_d0_V2*I_new_PINN[-1]/newI_ub_d0_V2[0]
newH_ub_d0_V2 = newH_ub_d0_V2*H_new_PINN[-1]/newH_ub_d0_V2[0]
newD_ub_d0_V2 = newD_ub_d0_V2*D_new_PINN[-1]/newD_ub_d0_V2[0] 

Sol_lb_d0_V2 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (-1,Pert0))  
S_lb_d0_V2 = Sol_lb_d0_V2[:,0]   
E_lb_d0_V2 = Sol_lb_d0_V2[:,1]   
PreS_lb_d0_V2 = Sol_lb_d0_V2[:,2]   
I_lb_d0_V2 = Sol_lb_d0_V2[:,3]   
J_lb_d0_V2 = Sol_lb_d0_V2[:,4]   
D_lb_d0_V2 = Sol_lb_d0_V2[:,5]   
H_lb_d0_V2 = Sol_lb_d0_V2[:,6]   
R_lb_d0_V2 = Sol_lb_d0_V2[:,7]   
sumI_lb_d0_V2 = Sol_lb_d0_V2[:,8] 
sumH_lb_d0_V2 = Sol_lb_d0_V2[:,9]  
newI_lb_d0_V2 = np.diff(sumI_lb_d0_V2)
newH_lb_d0_V2 = np.diff(sumH_lb_d0_V2)
newD_lb_d0_V2 = np.diff(D_lb_d0_V2) 
newI_lb_d0_V2 = newI_lb_d0_V2*I_new_PINN[-1]/newI_lb_d0_V2[0]
newH_lb_d0_V2 = newH_lb_d0_V2*H_new_PINN[-1]/newH_lb_d0_V2[0]
newD_lb_d0_V2 = newD_lb_d0_V2*D_new_PINN[-1]/newD_lb_d0_V2[0] 

#%%
Pert1 = 0.20 
Sol_ub_d1_V1 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (1,Pert1))  
S_ub_d1_V1 = Sol_ub_d1_V1[:,0]   
E_ub_d1_V1 = Sol_ub_d1_V1[:,1]      
PreS_ub_d1_V1 = Sol_ub_d1_V1[:,2] 
I_ub_d1_V1 = Sol_ub_d1_V1[:,3]   
J_ub_d1_V1 = Sol_ub_d1_V1[:,4]   
D_ub_d1_V1 = Sol_ub_d1_V1[:,5]   
H_ub_d1_V1 = Sol_ub_d1_V1[:,6]   
R_ub_d1_V1 = Sol_ub_d1_V1[:,7]   
sumI_ub_d1_V1 = Sol_ub_d1_V1[:,8] 
sumH_ub_d1_V1 = Sol_ub_d1_V1[:,9]  
newI_ub_d1_V1 = np.diff(sumI_ub_d1_V1)
newH_ub_d1_V1 = np.diff(sumH_ub_d1_V1)
newD_ub_d1_V1 = np.diff(D_ub_d1_V1)  
newI_ub_d1_V1 = newI_ub_d1_V1*I_new_PINN[-1]/newI_ub_d1_V1[0]
newH_ub_d1_V1 = newH_ub_d1_V1*H_new_PINN[-1]/newH_ub_d1_V1[0]
newD_ub_d1_V1 = newD_ub_d1_V1*D_new_PINN[-1]/newD_ub_d1_V1[0] 

Sol_lb_d1_V1 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (-1,Pert1))  
S_lb_d1_V1 = Sol_lb_d1_V1[:,0]   
E_lb_d1_V1 = Sol_lb_d1_V1[:,1]   
PreS_lb_d1_V1 = Sol_lb_d1_V1[:,2]   
I_lb_d1_V1 = Sol_lb_d1_V1[:,3]   
J_lb_d1_V1 = Sol_lb_d1_V1[:,4]   
D_lb_d1_V1 = Sol_lb_d1_V1[:,5]   
H_lb_d1_V1 = Sol_lb_d1_V1[:,6]   
R_lb_d1_V1 = Sol_lb_d1_V1[:,7]   
sumI_lb_d1_V1 = Sol_lb_d1_V1[:,8] 
sumH_lb_d1_V1 = Sol_lb_d1_V1[:,9]  
newI_lb_d1_V1 = np.diff(sumI_lb_d1_V1)
newH_lb_d1_V1 = np.diff(sumH_lb_d1_V1)
newD_lb_d1_V1 = np.diff(D_lb_d1_V1)  
newI_lb_d1_V1 = newI_lb_d1_V1*I_new_PINN[-1]/newI_lb_d1_V1[0]
newH_lb_d1_V1 = newH_lb_d1_V1*H_new_PINN[-1]/newH_lb_d1_V1[0]
newD_lb_d1_V1 = newD_lb_d1_V1*D_new_PINN[-1]/newD_lb_d1_V1[0] 


Sol_ub_d1_V2 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (1,Pert1))  
S_ub_d1_V2 = Sol_ub_d1_V2[:,0]   
E_ub_d1_V2 = Sol_ub_d1_V2[:,1]   
PreS_ub_d1_V2 = Sol_ub_d1_V2[:,2]   
I_ub_d1_V2 = Sol_ub_d1_V2[:,3]   
J_ub_d1_V2 = Sol_ub_d1_V2[:,4]   
D_ub_d1_V2 = Sol_ub_d1_V2[:,5]   
H_ub_d1_V2 = Sol_ub_d1_V2[:,6]   
R_ub_d1_V2 = Sol_ub_d1_V2[:,7]   
sumI_ub_d1_V2 = Sol_ub_d1_V2[:,8] 
sumH_ub_d1_V2 = Sol_ub_d1_V2[:,9]  
newI_ub_d1_V2 = np.diff(sumI_ub_d1_V2)
newH_ub_d1_V2 = np.diff(sumH_ub_d1_V2)
newD_ub_d1_V2 = np.diff(D_ub_d1_V2)  
newI_ub_d1_V2 = newI_ub_d1_V2*I_new_PINN[-1]/newI_ub_d1_V2[0]
newH_ub_d1_V2 = newH_ub_d1_V2*H_new_PINN[-1]/newH_ub_d1_V2[0]
newD_ub_d1_V2 = newD_ub_d1_V2*D_new_PINN[-1]/newD_ub_d1_V2[0] 

Sol_lb_d1_V2 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (-1,Pert1))  
S_lb_d1_V2 = Sol_lb_d1_V2[:,0]   
E_lb_d1_V2 = Sol_lb_d1_V2[:,1]   
PreS_lb_d1_V2 = Sol_lb_d1_V2[:,2]   
I_lb_d1_V2 = Sol_lb_d1_V2[:,3]   
J_lb_d1_V2 = Sol_lb_d1_V2[:,4]   
D_lb_d1_V2 = Sol_lb_d1_V2[:,5]   
H_lb_d1_V2 = Sol_lb_d1_V2[:,6]   
R_lb_d1_V2 = Sol_lb_d1_V2[:,7]   
sumI_lb_d1_V2 = Sol_lb_d1_V2[:,8] 
sumH_lb_d1_V2 = Sol_lb_d1_V2[:,9]  
newI_lb_d1_V2 = np.diff(sumI_lb_d1_V2)
newH_lb_d1_V2 = np.diff(sumH_lb_d1_V2)
newD_lb_d1_V2 = np.diff(D_lb_d1_V2) 
newI_lb_d1_V2 = newI_lb_d1_V2*I_new_PINN[-1]/newI_lb_d1_V2[0]
newH_lb_d1_V2 = newH_lb_d1_V2*H_new_PINN[-1]/newH_lb_d1_V2[0]
newD_lb_d1_V2 = newD_lb_d1_V2*D_new_PINN[-1]/newD_lb_d1_V2[0] 

#%% 
Sol_mean_V1 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (0,0))  
S_mean_V1 = Sol_mean_V1[:,0]   
E_mean_V1 = Sol_mean_V1[:,1]    
PreS_mean_V1 = Sol_mean_V1[:,2]   
I_mean_V1 = Sol_mean_V1[:,3]   
J_mean_V1 = Sol_mean_V1[:,4]   
D_mean_V1 = Sol_mean_V1[:,5]   
H_mean_V1 = Sol_mean_V1[:,6]   
R_mean_V1 = Sol_mean_V1[:,7]   
sumI_mean_V1 = Sol_mean_V1[:,8] 
sumH_mean_V1 = Sol_mean_V1[:,9]  
newI_mean_V1 = np.diff(sumI_mean_V1)
newH_mean_V1 = np.diff(sumH_mean_V1)
newD_mean_V1 = np.diff(D_mean_V1)  
newI_mean_V1 = newI_mean_V1*I_new_PINN[-1]/newI_mean_V1[0]
newH_mean_V1 = newH_mean_V1*H_new_PINN[-1]/newH_mean_V1[0]
newD_mean_V1 = newD_mean_V1*D_new_PINN[-1]/newD_mean_V1[0]
 
Sol_mean_V2 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (0,0))  
S_mean_V2 = Sol_mean_V2[:,0]   
E_mean_V2 = Sol_mean_V2[:,1]    
PreS_mean_V2 = Sol_mean_V2[:,2]   
I_mean_V2 = Sol_mean_V2[:,3]   
J_mean_V2 = Sol_mean_V2[:,4]   
D_mean_V2 = Sol_mean_V2[:,5]   
H_mean_V2 = Sol_mean_V2[:,6]   
R_mean_V2 = Sol_mean_V2[:,7]   
sumI_mean_V2 = Sol_mean_V2[:,8] 
sumH_mean_V2 = Sol_mean_V2[:,9]  
newI_mean_V2 = np.diff(sumI_mean_V2)
newH_mean_V2 = np.diff(sumH_mean_V2)
newD_mean_V2 = np.diff(D_mean_V2)  
newI_mean_V2 = newI_mean_V2*I_new_PINN[-1]/newI_mean_V2[0]
newH_mean_V2 = newH_mean_V2*H_new_PINN[-1]/newH_mean_V2[0]
newD_mean_V2 = newD_mean_V2*D_new_PINN[-1]/newD_mean_V2[0]

#%%
######################################################################
######################################################################
############################# Save the results ###############################
######################################################################
###################################################################### 
#%% 
#saver
current_directory = os.getcwd()
relative_path = '/Model2/Prediction-Results-'+dt_string+'/'
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    

np.savetxt(save_results_to + 'S_mean_V1.txt', S_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_mean_V1.txt', E_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'PreS_mean_V1.txt', PreS_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_mean_V1.txt', I_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_mean_V1.txt', J_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_mean_V1.txt', D_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_mean_V1.txt', H_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_mean_V1.txt', R_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'newI_mean_V1.txt', newI_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'newH_mean_V1.txt', newH_mean_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'newD_mean_V1.txt', newD_mean_V1.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_ub_d0_V1.txt', S_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_ub_d0_V1.txt', E_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'PreS_ub_d0_V1.txt', PreS_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_ub_d0_V1.txt', I_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_ub_d0_V1.txt', J_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_ub_d0_V1.txt', D_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_ub_d0_V1.txt', H_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_ub_d0_V1.txt', R_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'newI_ub_d0_V1.txt', newI_ub_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'newH_ub_d0_V1.txt', newH_ub_d0_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'newD_ub_d0_V1.txt', newD_ub_d0_V1.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_lb_d0_V1.txt', S_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_lb_d0_V1.txt', E_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'PreS_lb_d0_V1.txt', PreS_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_lb_d0_V1.txt', I_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_lb_d0_V1.txt', J_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_lb_d0_V1.txt', D_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_lb_d0_V1.txt', H_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_lb_d0_V1.txt', R_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'newI_lb_d0_V1.txt', newI_lb_d0_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'newH_lb_d0_V1.txt', newH_lb_d0_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'newD_lb_d0_V1.txt', newD_lb_d0_V1.reshape((-1,1))) 


np.savetxt(save_results_to + 'S_mean_V2.txt', S_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_mean_V2.txt', E_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'PreS_mean_V2.txt', PreS_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_mean_V2.txt', I_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_mean_V2.txt', J_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_mean_V2.txt', D_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_mean_V2.txt', H_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_mean_V2.txt', R_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'newI_mean_V2.txt', newI_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'newH_mean_V2.txt', newH_mean_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'newD_mean_V2.txt', newD_mean_V2.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_ub_d0_V2.txt', S_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_ub_d0_V2.txt', E_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'PreS_ub_d0_V2.txt', PreS_ub_d0_V2.reshape((-1,1)))  
np.savetxt(save_results_to + 'I_ub_d0_V2.txt', I_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_ub_d0_V2.txt', J_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_ub_d0_V2.txt', D_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_ub_d0_V2.txt', H_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_ub_d0_V2.txt', R_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'newI_ub_d0_V2.txt', newI_ub_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'newH_ub_d0_V2.txt', newH_ub_d0_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'newD_ub_d0_V2.txt', newD_ub_d0_V2.reshape((-1,1)))  

np.savetxt(save_results_to + 'S_lb_d0_V2.txt', S_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'E_lb_d0_V2.txt', E_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'PreS_lb_d0_V2.txt', PreS_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'I_lb_d0_V2.txt', I_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'J_lb_d0_V2.txt', J_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'D_lb_d0_V2.txt', D_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'H_lb_d0_V2.txt', H_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'R_lb_d0_V2.txt', R_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'newI_lb_d0_V2.txt', newI_lb_d0_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'newH_lb_d0_V2.txt', newH_lb_d0_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'newD_lb_d0_V2.txt', newD_lb_d0_V2.reshape((-1,1))) 

#%%
######################################################################
############################# Plotting ###############################
######################################################################  

plt.rc('font', size=60)
intervals=[1,2,1]

#%%
plt.rc('font', size=40) 
# V1 = data_frame['Daily number of first vaccine doses administered (includes single-dose vaccines)']
# V2 = data_frame['Daily number of second vaccine doses administered (only applicable for two-dose vaccines)']
# V1 = V1[291:].to_numpy(dtype=np.float64)
# V2 = V2[291:].to_numpy(dtype=np.float64)
# V_eff = 0.52*(V1-V2)+0.95*V2
# V_eff = np.convolve(V_eff, np.ones(7), 'valid') / 7

# V_pred_1 = np.asarray([V_1(time) for time in t_pred], dtype=np.float64)
# V_pred_2 = np.asarray([V_2(time) for time in t_pred], dtype=np.float64)
# # plt.plot(V_eff)
# # plt.plot(V1)
# # plt.plot(V2)

# fig, ax = plt.subplots() 
# ax.bar(date_total[290:].flatten(), V1[6:], label='dose 1 vacc.')
# ax.bar(date_total[290:].flatten(), V2[6:], label='dose 2 vacc.')
# ax.plot(date_total[290:].flatten(), V_eff, 'k-', lw=5, label='current eff. vacc')
# ax.plot(data_pred.flatten(), V_pred_1, 'r-', lw=5, label='future eff. vacc.')
# ax.plot(data_pred.flatten(), V_pred_2, 'r--', lw=5, label='future eff. vacc. (higher rate)')


# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
# plt.xticks(rotation=30)
# ax.legend(fontsize=35, ncol = 1, loc = 'best')
# ax.tick_params(axis='x', labelsize = 40)
# ax.tick_params(axis='y', labelsize = 40)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# plt.rc('font', size=40)
# # ax.grid(True)
# # ax.set_xlabel('Date', fontsize = font) 
# ax.set_ylabel('$vaccination$', fontsize = 80) 
# fig.set_size_inches(w=25, h=12.5) 
# plt.savefig(save_results_to +'Vaccination.pdf', dpi=300)  
# plt.savefig(save_results_to +'Vaccination.png', dpi=300)   


#%%
#BetaI curve   
beta_pred_0 = np.array([BetaI_PINN[-1] for i in range(data_pred[1:].shape[0])])

fig, ax = plt.subplots() 
ax.plot(data_mean, BetaI_PINN, 'k-', lw=4, label='PINN-Training')   
ax.plot(data_pred[1:].flatten(), beta_pred_0, 'm--', lw=4, label='Prediction-mean')   
plt.fill_between(data_pred[1:].flatten(), \
                 beta_pred_0*(1.1), \
                 beta_pred_0*(0.9), \
                 facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')   

plt.fill_between(data_pred[1:].flatten(), \
                 beta_pred_0*(1.2), \
                 beta_pred_0*(0.8), \
                 facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')   


ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
# ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
plt.rc('font', size=40)
ax.grid(True)
# ax.set_xlabel('Date', fontsize = font) 
ax.set_ylabel(r'$\beta_{I}$', fontsize = 80) 
fig.set_size_inches(w=25, h=12.5) 
plt.savefig(save_results_to +'BetaI.pdf', dpi=300)  
plt.savefig(save_results_to +'BetaI.png', dpi=300)  

#%% 
#New infectious 
for i in [1,2]:
    fig, ax = plt.subplots() 
    plt.fill_between(data_pred[:-1].flatten(), \
                      newI_lb_d0_V1.flatten(), newI_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newI_lb_d1_V1.flatten(), newI_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newI_lb_d0_V2.flatten(), newI_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newI_lb_d1_V2.flatten(), newI_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred[:-1], newI_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred[:-1], newI_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(date_total, I_new_star, 'ro', lw=4, markersize=8, label='Data-7davg')
        # ax.plot(data_mean[:-1], I_new_PINN, 'k-', lw=4, label='PINN-Training')   
        ax.plot(data_mean[1:], I_new_PINN, 'k-', lw=4, label='PINN-Training')   
        ax.set_xlim(date_total[18,:],data_pred[-1])

    if i==2:
        ax.plot(data_pred[:-1], newI_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred[:-1], newI_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'upper left')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('daily infectious cases ($\mathbf{I}^{new}$)', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'new_cases_7davg.pdf', dpi=300) 
        plt.savefig(save_results_to + 'new_cases_7davg.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'new_cases_7davg_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'new_cases_7davg_zoom.png', dpi=300)  

#%% 
#New hospitalized 
for i in [1,2]:
    fig, ax = plt.subplots() 
    plt.fill_between(data_pred[:-1].flatten(), \
                      newH_lb_d0_V1.flatten(), newH_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newH_lb_d1_V1.flatten(), newH_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newH_lb_d0_V2.flatten(), newH_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newH_lb_d1_V2.flatten(), newH_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    if i==1:
        ax.plot(data_pred[:-1], newH_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred[:-1], newH_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)') 
        ax.plot(data_mean[1:], H_new_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred[:-1], newH_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred[:-1], newH_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('daily hospitalized cases ($\mathbf{H}^{new}$)', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'new_hospitalized_7davg.pdf', dpi=300) 
        plt.savefig(save_results_to + 'new_hospitalized_7davg.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'new_hospitalized_7davg_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'new_hospitalized_7davg_zoom.png', dpi=300)  

#%% 
#New death 
for i in [1,2]:
    fig, ax = plt.subplots() 
    plt.fill_between(data_pred[:-1].flatten(), \
                      newD_lb_d0_V1.flatten(), newD_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newD_lb_d1_V1.flatten(), newD_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newD_lb_d0_V2.flatten(), newD_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      newD_lb_d1_V2.flatten(), newD_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred[:-1], newD_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred[:-1], newD_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(date_total, D_new_star, 'ro', lw=4, markersize=8, label='Data-7davg')
        # ax.plot(data_mean[:-1], D_new_PINN, 'k-', lw=4, label='PINN-Training')   
        ax.plot(data_mean[1:], D_new_PINN, 'k-', lw=4, label='PINN-Training')    
        ax.set_xlim(date_total[18,:],data_pred[-1])

    if i==2:
        ax.plot(data_pred[:-1], newD_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred[:-1], newD_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('daily death cases ($\mathbf{D}^{new}$)', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'new_death_7davg.pdf', dpi=300) 
        plt.savefig(save_results_to + 'new_death_7davg.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'new_death_7davg_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'new_death_7davg_zoom.png', dpi=300)  

#%% 
#Current Suspectious
for i in [1,2]:
    fig, ax = plt.subplots() 
    plt.fill_between(data_pred.flatten(), \
                      S_lb_d0_V1.flatten(), S_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      S_lb_d1_V1.flatten(), S_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      S_lb_d0_V2.flatten(), S_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      S_lb_d1_V2.flatten(), S_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, S_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, S_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, S_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, S_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, S_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{S}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Suspectious.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Suspectious.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Suspectious_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Suspectious_zoom.png', dpi=300)  

#%% 
#Current Exposed
for i in [1,2]:
    fig, ax = plt.subplots() 
    plt.fill_between(data_pred.flatten(), \
                      E_lb_d0_V1.flatten(), E_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      E_lb_d1_V1.flatten(), E_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      E_lb_d0_V2.flatten(), E_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      E_lb_d1_V2.flatten(), E_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, E_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, E_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, E_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, E_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, E_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{E}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Exposed.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Exposed.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Exposed_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Exposed_zoom.png', dpi=300)  

#%% 
#Current PreSymptomatic
for i in [1,2]:
    fig, ax = plt.subplots() 
    plt.fill_between(data_pred.flatten(), \
                      PreS_lb_d0_V1.flatten(), PreS_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      PreS_lb_d1_V1.flatten(), PreS_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      PreS_lb_d0_V2.flatten(), PreS_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      PreS_lb_d1_V2.flatten(), PreS_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, PreS_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, PreS_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, PreS_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, PreS_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, PreS_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{P}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Presymptomatic.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Presymptomatic.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Presymptomatic_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Presymptomatic_zoom.png', dpi=300)  

#%% 
#Current infectious
for i in [1,2]:
    fig, ax = plt.subplots()  
    plt.fill_between(data_pred.flatten(), \
                      I_lb_d0_V1.flatten(), I_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      I_lb_d1_V1.flatten(), I_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      I_lb_d0_V2.flatten(), I_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      I_lb_d1_V2.flatten(), I_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, I_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, I_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, I_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, I_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, I_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    
    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{I}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Infectious.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Infectious.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Infectious_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Infectious_zoom.png', dpi=300)  

#%% 
#Current asymptomatic
for i in [1,2]:
    fig, ax = plt.subplots()  
    plt.fill_between(data_pred.flatten(), \
                      J_lb_d0_V1.flatten(), J_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      J_lb_d1_V1.flatten(), J_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      J_lb_d0_V2.flatten(), J_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      J_lb_d1_V2.flatten(), J_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, J_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, J_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, J_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, J_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, J_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    
    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{J}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Asymptomatic.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Asymptomatic.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Asymptomatic_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Asymptomatic_zoom.png', dpi=300)  

#%% 
#Current death
for i in [1,2]:
    fig, ax = plt.subplots()  
    plt.fill_between(data_pred.flatten(), \
                      D_lb_d0_V1.flatten(), D_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      D_lb_d1_V1.flatten(), D_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      D_lb_d0_V2.flatten(), D_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      D_lb_d1_V2.flatten(), D_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, D_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, D_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, D_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, D_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, D_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    
    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{D}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Death.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Death.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Death_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Death_zoom.png', dpi=300)  

#%% 
#Current hospitalized
for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred.flatten(), \
                      H_lb_d0_V1.flatten(), H_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      H_lb_d1_V1.flatten(), H_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      H_lb_d0_V2.flatten(), H_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      H_lb_d1_V2.flatten(), H_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, H_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, H_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')       
        ax.plot(date_total, H_star, 'ro', lw=4, markersize=8, label='Data-7davg') 
        ax.plot(data_mean, H_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, H_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, H_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    
    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{H}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Hospitalized.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Hospitalized.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Hospitalized_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Hospitalized_zoom.png', dpi=300)  

#%% 
#Current removed
for i in [1,2]:
    fig, ax = plt.subplots()  
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d0_V1.flatten(), R_ub_d0_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d1_V1.flatten(), R_ub_d1_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d0_V2.flatten(), R_ub_d0_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      R_lb_d1_V2.flatten(), R_ub_d1_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, R_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, R_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, R_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, R_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, R_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    
    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbf{R}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Recovered.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Recovered.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Recovered_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Recovered_zoom.png', dpi=300)  

