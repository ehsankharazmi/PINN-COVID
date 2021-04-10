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
dt_string =  '03-13'

# Load Data 
data_frame = pandas.read_csv('Data/data-by-day.csv')  
I_new_star = data_frame['CASE_COUNT'] #T x 1 array 
H_new_star = data_frame['HOSPITALIZED_COUNT'] #T x 1 array
D_new_star = data_frame['DEATH_COUNT'] #T x 1 array 

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

first_date = '2020-03-06' #first_date[6:]+'-'+first_date[0:2]+'-'+first_date[3:5]
last_date = '2021-03-11' #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])+1)
first_date_pred = '2021-03-09' #last_date[6:]+'-'+last_date[0:2]+'-'+str(int(last_date[3:5])-1)
last_date_pred = '2021-07-01'

date_total = np.arange(first_date, last_date, dtype='datetime64[D]')[:,None] 
data_mean = np.arange(first_date, last_date, dtype='datetime64[D]')[:,None]
data_pred = np.arange(first_date_pred, last_date_pred, dtype='datetime64[D]')[:,None] 


sf = 1e-4

# load data
BetaI_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/BetaI_pred_mean.txt') 
p_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/p_pred_mean.txt')
q_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/q_pred_mean.txt') 
t_mean = np.arange(len(BetaI_PINN))

S_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/S_pred_mean.txt') 
S_PINN = S_PINN/sf
I_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/I_pred_mean.txt') 
I_PINN = I_PINN/sf
J_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/J_pred_mean.txt') 
J_PINN = J_PINN/sf
H_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/H_pred_mean.txt') 
H_PINN = H_PINN/sf
D_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/D_pred_mean.txt') 
D_PINN = D_PINN/sf
R_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/R_pred_mean.txt') 
R_PINN = R_PINN/sf
I_sum_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/I_sum_pred_mean.txt') 
I_sum_PINN = I_sum_PINN/sf
H_sum_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/H_sum_pred_mean.txt') 
H_sum_PINN = H_sum_PINN/sf
I_new_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/I_new_pred_mean.txt') 
I_new_PINN = I_new_PINN/sf
H_new_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/H_new_pred_mean.txt') 
H_new_PINN = H_new_PINN/sf
D_new_PINN = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/D_new_pred_mean.txt') 
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
I_init = float(I_PINN[-2]) 
J_init = float(J_PINN[-2])  
D_init = float(D_PINN[-2]) 
H_init = float(H_PINN[-2]) 
R_init = float(R_PINN[-2]) 
I_sum_init = float(I_sum_PINN[-2]) 
D_sum_init = float(D_PINN[-2]) 
H_sum_init = float(H_sum_PINN[-2])  
U_init = [S_init, I_init, J_init, D_init, H_init, R_init, I_sum_init, H_sum_init]

#Parameters      
#Parameters     
eps1 = 0.75
eps2 = 0.0
delta = 0.6
alpha = 1.0/5.2
Gamma = 1.0/6.0
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
    if Dt <= 300:
        return 0 
    elif Dt<350:
        return (Dt-300)*600 
    else:
        return 30000 
    
def V_2(Dt):
    # return 0
    if Dt <= 300:
        return 0 
    elif Dt<350:
        return (Dt-300)*1200 
    else:
        return 60000 

#ODEs  
def ODEs_mean_1(X, t, xi, Pert): 
    S, I, J,  D, H, R, sumI, sumH = X  
    dSdt = -(BetaI_PINN[-1]  * (1+xi*Pert) *(I+eps1*J+eps2*H)/N)*S - V_1(t)/N*S
    delay = (BetaI_PINN[-1]  * (1+xi*Pert) *(I+eps1*J+eps2*H)/N)*S  
    dIdt = delta*delay - Gamma*I 
    dJdt = (1-delta)*delay - gammaA*J
    dDdt = (q_mean*phiD)*H
    dHdt = (p_mean*Gamma)*I - (q_mean*phiD) * H - ((1-q_mean)*phiR) * H
    dRdt = gammaA*J + ((1-p_mean)*Gamma)*I + ((1-q_mean)*phiR)*H + V_1(t)/N*S
    dsumIdt = delta*delay
    dsumHdt = (p_mean*Gamma)*I   
    return [dSdt, dIdt, dJdt, dDdt, dHdt, dRdt, dsumIdt, dsumHdt] 

def ODEs_mean_2(X, t, xi, Pert):  
    S, I, J,  D, H, R, sumI, sumH = X  
    dSdt = -(BetaI_PINN[-1]  * (1+xi*Pert) *(I+eps1*J+eps2*H)/N)*S - V_2(t)/N*S
    delay = (BetaI_PINN[-1]  * (1+xi*Pert) *(I+eps1*J+eps2*H)/N)*S 
    dIdt = delta*delay - Gamma*I 
    dJdt = (1-delta)*delay - gammaA*J
    dDdt = (q_mean*phiD)*H
    dHdt = (p_mean*Gamma)*I - (q_mean*phiD) * H - ((1-q_mean)*phiR) * H
    dRdt = gammaA*J + ((1-p_mean)*Gamma)*I + ((1-q_mean)*phiR)*H + V_2(t)/N*S
    dsumIdt = delta*delay
    dsumHdt = (p_mean*Gamma)*I   
    return [dSdt, dIdt, dJdt, dDdt, dHdt, dRdt, dsumIdt, dsumHdt]   

#%%
#sample points in [-1, 1] as Guass-Labo
N_sample = 10
[Xi,Weights] = np.polynomial.legendre.leggauss(N_sample)

#Solver 
Pert0 = 0.175
Pert1 = 0.35

# BetaI_pred_d0 = [] 
# BetaI_pred_d1 = [] 

Sol_S_d0 = [] 
Sol_I_d0 = []
Sol_J_d0 = [] 
Sol_D_d0 = []
Sol_H_d0 = []
Sol_R_d0 = []
Sol_newI_d0 = []
Sol_newH_d0 = []
Sol_newD_d0 = []

Sol_S_d1 = [] 
Sol_I_d1 = []
Sol_J_d1 = [] 
Sol_D_d1 = []
Sol_H_d1 = []
Sol_R_d1 = []
Sol_newI_d1 = []
Sol_newH_d1 = []
Sol_newD_d1 = []
for n in range(N_sample):
    xi = Xi[n].reshape([1,1]) 
    #No Vaccine
    Sol_d0 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (xi,Pert0))  
    Sol_S_n_d0 = Sol_d0[:,0]   
    Sol_I_n_d0 = Sol_d0[:,1]   
    Sol_J_n_d0 = Sol_d0[:,2]   
    Sol_D_n_d0 = Sol_d0[:,3]   
    Sol_H_n_d0 = Sol_d0[:,4]   
    Sol_R_n_d0 = Sol_d0[:,5]   
    Sol_sumI_n_d0 = Sol_d0[:,6]   
    Sol_sumH_n_d0 = Sol_d0[:,7]   
    Sol_S_d0.append(Sol_S_n_d0.reshape([len(Sol_S_n_d0),1])) 
    Sol_I_d0.append(Sol_I_n_d0.reshape([len(Sol_I_n_d0),1]))
    Sol_J_d0.append(Sol_J_n_d0.reshape([len(Sol_J_n_d0),1])) 
    Sol_D_d0.append(Sol_D_n_d0.reshape([len(Sol_D_n_d0),1]))
    Sol_H_d0.append(Sol_H_n_d0.reshape([len(Sol_H_n_d0),1]))
    Sol_R_d0.append(Sol_R_n_d0.reshape([len(Sol_R_n_d0),1])) 
    Sol_newI_n_d0 = np.diff(Sol_sumI_n_d0)
    Sol_newH_n_d0 = np.diff(Sol_sumH_n_d0)
    Sol_newD_n_d0 = np.diff(Sol_D_n_d0)  
    Sol_newI_d0.append(Sol_newI_n_d0.reshape([len(Sol_newI_n_d0),1])) 
    Sol_newH_d0.append(Sol_newH_n_d0.reshape([len(Sol_newH_n_d0),1])) 
    Sol_newD_d0.append(Sol_newD_n_d0.reshape([len(Sol_newD_n_d0),1]))  
    
    #With Vaccine
    Sol_d1 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (xi,Pert1))  
    Sol_S_n_d1 = Sol_d1[:,0]   
    Sol_I_n_d1 = Sol_d1[:,1]   
    Sol_J_n_d1 = Sol_d1[:,2]   
    Sol_D_n_d1 = Sol_d1[:,3]   
    Sol_H_n_d1 = Sol_d1[:,4]   
    Sol_R_n_d1 = Sol_d1[:,5]   
    Sol_sumI_n_d1 = Sol_d1[:,6]   
    Sol_sumH_n_d1 = Sol_d1[:,7]  
    Sol_S_d1.append(Sol_S_n_d1.reshape([len(Sol_S_n_d1),1])) 
    Sol_I_d1.append(Sol_I_n_d1.reshape([len(Sol_I_n_d1),1]))
    Sol_J_d1.append(Sol_J_n_d1.reshape([len(Sol_J_n_d1),1])) 
    Sol_D_d1.append(Sol_D_n_d1.reshape([len(Sol_D_n_d1),1]))
    Sol_H_d1.append(Sol_H_n_d1.reshape([len(Sol_H_n_d1),1]))
    Sol_R_d1.append(Sol_R_n_d1.reshape([len(Sol_R_n_d1),1])) 
    Sol_newI_n_d1 = np.diff(Sol_sumI_n_d1)
    Sol_newH_n_d1 = np.diff(Sol_sumH_n_d1)
    Sol_newD_n_d1 = np.diff(Sol_D_n_d1)  
    Sol_newI_d1.append(Sol_newI_n_d1.reshape([len(Sol_newI_n_d1),1])) 
    Sol_newH_d1.append(Sol_newH_n_d1.reshape([len(Sol_newH_n_d1),1])) 
    Sol_newD_d1.append(Sol_newD_n_d1.reshape([len(Sol_newD_n_d1),1]))   

Sol_S1_d0 = np.asarray(Sol_S_d0, order= 'F')
Sol_S2_d0 = Sol_S1_d0.reshape([N_sample, len(t_pred)])
Sol_S_d0_mat = np.transpose(Sol_S2_d0) 
Sol_S_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_S_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_S_d0_mat, Weights)*0.5))  
Sol_I1_d0 = np.asarray(Sol_I_d0, order= 'F')
Sol_I2_d0 = Sol_I1_d0.reshape([N_sample, len(t_pred)])
Sol_I_d0_mat = np.transpose(Sol_I2_d0) 
Sol_I_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_I_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_I_d0_mat, Weights)*0.5))   
Sol_J1_d0 = np.asarray(Sol_J_d0, order= 'F')
Sol_J2_d0 = Sol_J1_d0.reshape([N_sample, len(t_pred)])
Sol_J_d0_mat = np.transpose(Sol_J2_d0) 
Sol_J_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_J_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_J_d0_mat, Weights)*0.5))  
Sol_D1_d0 = np.asarray(Sol_D_d0, order= 'F')
Sol_D2_d0 = Sol_D1_d0.reshape([N_sample, len(t_pred)])
Sol_D_d0_mat = np.transpose(Sol_D2_d0) 
Sol_D_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_D_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_D_d0_mat, Weights)*0.5))  
Sol_H1_d0 = np.asarray(Sol_H_d0, order= 'F')
Sol_H2_d0 = Sol_H1_d0.reshape([N_sample, len(t_pred)])
Sol_H_d0_mat = np.transpose(Sol_H2_d0) 
Sol_H_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_H_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_H_d0_mat, Weights)*0.5))   
Sol_R1_d0 = np.asarray(Sol_R_d0, order= 'F')
Sol_R2_d0 = Sol_R1_d0.reshape([N_sample, len(t_pred)])
Sol_R_d0_mat = np.transpose(Sol_R2_d0) 
Sol_R_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_R_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_R_d0_mat, Weights)*0.5))  
Sol_newI1_d0 = np.asarray(Sol_newI_d0, order= 'F')
Sol_newI2_d0 = Sol_newI1_d0.reshape([N_sample, len(t_pred)-1])
Sol_newI_d0_mat = np.transpose(Sol_newI2_d0) 
Sol_newI_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_newI_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newI_d0_mat, Weights)*0.5))   
Sol_newH1_d0 = np.asarray(Sol_newH_d0, order= 'F')
Sol_newH2_d0 = Sol_newH1_d0.reshape([N_sample, len(t_pred)-1])
Sol_newH_d0_mat = np.transpose(Sol_newH2_d0)  
Sol_newH_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_newH_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newH_d0_mat, Weights)*0.5))
Sol_newD1_d0 = np.asarray(Sol_newD_d0, order= 'F')
Sol_newD2_d0 = Sol_newD1_d0.reshape([N_sample, len(t_pred)-1])
Sol_newD_d0_mat = np.transpose(Sol_newD2_d0)  
Sol_newD_d0_std_V1 = np.sqrt(np.matmul(np.square(Sol_newD_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newD_d0_mat, Weights)*0.5)) 

Sol_S1_d1 = np.asarray(Sol_S_d1, order= 'F')
Sol_S2_d1 = Sol_S1_d1.reshape([N_sample, len(t_pred)])
Sol_S_d1_mat = np.transpose(Sol_S2_d1) 
Sol_S_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_S_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_S_d1_mat, Weights)*0.5))  
Sol_I1_d1 = np.asarray(Sol_I_d1, order= 'F')
Sol_I2_d1 = Sol_I1_d1.reshape([N_sample, len(t_pred)])
Sol_I_d1_mat = np.transpose(Sol_I2_d1) 
Sol_I_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_I_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_I_d1_mat, Weights)*0.5))   
Sol_J1_d1 = np.asarray(Sol_J_d1, order= 'F')
Sol_J2_d1 = Sol_J1_d1.reshape([N_sample, len(t_pred)])
Sol_J_d1_mat = np.transpose(Sol_J2_d1) 
Sol_J_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_J_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_J_d1_mat, Weights)*0.5))  
Sol_D1_d1 = np.asarray(Sol_D_d1, order= 'F')
Sol_D2_d1 = Sol_D1_d1.reshape([N_sample, len(t_pred)])
Sol_D_d1_mat = np.transpose(Sol_D2_d1) 
Sol_D_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_D_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_D_d1_mat, Weights)*0.5))  
Sol_H1_d1 = np.asarray(Sol_H_d1, order= 'F')
Sol_H2_d1 = Sol_H1_d1.reshape([N_sample, len(t_pred)])
Sol_H_d1_mat = np.transpose(Sol_H2_d1) 
Sol_H_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_H_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_H_d1_mat, Weights)*0.5))   
Sol_R1_d1 = np.asarray(Sol_R_d1, order= 'F')
Sol_R2_d1 = Sol_R1_d1.reshape([N_sample, len(t_pred)])
Sol_R_d1_mat = np.transpose(Sol_R2_d1) 
Sol_R_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_R_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_R_d1_mat, Weights)*0.5))  
Sol_newI1_d1 = np.asarray(Sol_newI_d1, order= 'F')
Sol_newI2_d1 = Sol_newI1_d1.reshape([N_sample, len(t_pred)-1])
Sol_newI_d1_mat = np.transpose(Sol_newI2_d1) 
Sol_newI_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_newI_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newI_d1_mat, Weights)*0.5))    
Sol_newH1_d1 = np.asarray(Sol_newH_d1, order= 'F')
Sol_newH2_d1 = Sol_newH1_d1.reshape([N_sample, len(t_pred)-1])
Sol_newH_d1_mat = np.transpose(Sol_newH2_d1)    
Sol_newH_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_newH_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newH_d1_mat, Weights)*0.5)) 
Sol_newD1_d1 = np.asarray(Sol_newD_d1, order= 'F')
Sol_newD2_d1 = Sol_newD1_d1.reshape([N_sample, len(t_pred)-1])
Sol_newD_d1_mat = np.transpose(Sol_newD2_d1)   
Sol_newD_d1_std_V1 = np.sqrt(np.matmul(np.square(Sol_newD_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newD_d1_mat, Weights)*0.5))  

Sol_mean_V1 = odeint(ODEs_mean_1, U_init, t_pred.flatten(), args = (0,0))     
Sol_S_mean_V1 = Sol_mean_V1[:,0]
Sol_I_mean_V1 = Sol_mean_V1[:,1] 
Sol_J_mean_V1 = Sol_mean_V1[:,2] 
Sol_D_mean_V1 = Sol_mean_V1[:,3] 
Sol_H_mean_V1 = Sol_mean_V1[:,4] 
Sol_R_mean_V1 = Sol_mean_V1[:,5] 
Sol_sumI_mean_V1 = Sol_mean_V1[:,6] 
Sol_sumH_mean_V1 = Sol_mean_V1[:,7]     
Sol_newI_mean_V1 = np.diff(Sol_sumI_mean_V1)
Sol_newH_mean_V1 = np.diff(Sol_sumH_mean_V1)
Sol_newD_mean_V1 = np.diff(Sol_D_mean_V1)  
Diff_newI = I_new_PINN[-1]/Sol_newI_mean_V1[0] 
Diff_newH = H_new_PINN[-1]/Sol_newH_mean_V1[0] 
Diff_newD = D_new_PINN[-1]/Sol_newD_mean_V1[0]  
Sol_newI_mean_V1 = Sol_newI_mean_V1 * Diff_newI 
Sol_newH_mean_V1 = Sol_newH_mean_V1 * Diff_newH 
Sol_newD_mean_V1 = Sol_newD_mean_V1 * Diff_newD   

#%%
#sample points in [-1, 1] as Guass-Labo
N_sample = 10
[Xi,Weights] = np.polynomial.legendre.leggauss(N_sample)

#Solver 
Pert0 = 0.175
Pert1 = 0.35

# BetaI_pred_d0 = [] 
# BetaI_pred_d1 = [] 

Sol_S_d0 = [] 
Sol_I_d0 = []
Sol_J_d0 = [] 
Sol_D_d0 = []
Sol_H_d0 = []
Sol_R_d0 = []
Sol_newI_d0 = []
Sol_newH_d0 = []
Sol_newD_d0 = []

Sol_S_d1 = [] 
Sol_I_d1 = []
Sol_J_d1 = [] 
Sol_D_d1 = []
Sol_H_d1 = []
Sol_R_d1 = []
Sol_newI_d1 = []
Sol_newH_d1 = []
Sol_newD_d1 = []
for n in range(N_sample):
    xi = Xi[n].reshape([1,1]) 
    #No Vaccine 
    Sol_d0 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (xi,Pert0)) 
    Sol_S_n_d0 = Sol_d0[:,0]   
    Sol_I_n_d0 = Sol_d0[:,1]   
    Sol_J_n_d0 = Sol_d0[:,2]   
    Sol_D_n_d0 = Sol_d0[:,3]   
    Sol_H_n_d0 = Sol_d0[:,4]   
    Sol_R_n_d0 = Sol_d0[:,5]   
    Sol_sumI_n_d0 = Sol_d0[:,6]   
    Sol_sumH_n_d0 = Sol_d0[:,7]   
    Sol_S_d0.append(Sol_S_n_d0.reshape([len(Sol_S_n_d0),1])) 
    Sol_I_d0.append(Sol_I_n_d0.reshape([len(Sol_I_n_d0),1]))
    Sol_J_d0.append(Sol_J_n_d0.reshape([len(Sol_J_n_d0),1])) 
    Sol_D_d0.append(Sol_D_n_d0.reshape([len(Sol_D_n_d0),1]))
    Sol_H_d0.append(Sol_H_n_d0.reshape([len(Sol_H_n_d0),1]))
    Sol_R_d0.append(Sol_R_n_d0.reshape([len(Sol_R_n_d0),1])) 
    Sol_newI_n_d0 = np.diff(Sol_sumI_n_d0)
    Sol_newH_n_d0 = np.diff(Sol_sumH_n_d0)
    Sol_newD_n_d0 = np.diff(Sol_D_n_d0)  
    Sol_newI_d0.append(Sol_newI_n_d0.reshape([len(Sol_newI_n_d0),1])) 
    Sol_newH_d0.append(Sol_newH_n_d0.reshape([len(Sol_newH_n_d0),1])) 
    Sol_newD_d0.append(Sol_newD_n_d0.reshape([len(Sol_newD_n_d0),1]))  
    
    #With Vaccine 
    Sol_d1 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (xi,Pert1))  
    Sol_S_n_d1 = Sol_d1[:,0]   
    Sol_I_n_d1 = Sol_d1[:,1]   
    Sol_J_n_d1 = Sol_d1[:,2]   
    Sol_D_n_d1 = Sol_d1[:,3]   
    Sol_H_n_d1 = Sol_d1[:,4]   
    Sol_R_n_d1 = Sol_d1[:,5]   
    Sol_sumI_n_d1 = Sol_d1[:,6]   
    Sol_sumH_n_d1 = Sol_d1[:,7]  
    Sol_S_d1.append(Sol_S_n_d1.reshape([len(Sol_S_n_d1),1])) 
    Sol_I_d1.append(Sol_I_n_d1.reshape([len(Sol_I_n_d1),1]))
    Sol_J_d1.append(Sol_J_n_d1.reshape([len(Sol_J_n_d1),1])) 
    Sol_D_d1.append(Sol_D_n_d1.reshape([len(Sol_D_n_d1),1]))
    Sol_H_d1.append(Sol_H_n_d1.reshape([len(Sol_H_n_d1),1]))
    Sol_R_d1.append(Sol_R_n_d1.reshape([len(Sol_R_n_d1),1])) 
    Sol_newI_n_d1 = np.diff(Sol_sumI_n_d1)
    Sol_newH_n_d1 = np.diff(Sol_sumH_n_d1)
    Sol_newD_n_d1 = np.diff(Sol_D_n_d1)  
    Sol_newI_d1.append(Sol_newI_n_d1.reshape([len(Sol_newI_n_d1),1])) 
    Sol_newH_d1.append(Sol_newH_n_d1.reshape([len(Sol_newH_n_d1),1])) 
    Sol_newD_d1.append(Sol_newD_n_d1.reshape([len(Sol_newD_n_d1),1]))   

Sol_S1_d0 = np.asarray(Sol_S_d0, order= 'F')
Sol_S2_d0 = Sol_S1_d0.reshape([N_sample, len(t_pred)])
Sol_S_d0_mat = np.transpose(Sol_S2_d0) 
Sol_S_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_S_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_S_d0_mat, Weights)*0.5))  
Sol_I1_d0 = np.asarray(Sol_I_d0, order= 'F')
Sol_I2_d0 = Sol_I1_d0.reshape([N_sample, len(t_pred)])
Sol_I_d0_mat = np.transpose(Sol_I2_d0) 
Sol_I_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_I_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_I_d0_mat, Weights)*0.5))   
Sol_J1_d0 = np.asarray(Sol_J_d0, order= 'F')
Sol_J2_d0 = Sol_J1_d0.reshape([N_sample, len(t_pred)])
Sol_J_d0_mat = np.transpose(Sol_J2_d0) 
Sol_J_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_J_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_J_d0_mat, Weights)*0.5))   
Sol_D1_d0 = np.asarray(Sol_D_d0, order= 'F')
Sol_D2_d0 = Sol_D1_d0.reshape([N_sample, len(t_pred)])
Sol_D_d0_mat = np.transpose(Sol_D2_d0) 
Sol_D_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_D_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_D_d0_mat, Weights)*0.5))  
Sol_H1_d0 = np.asarray(Sol_H_d0, order= 'F')
Sol_H2_d0 = Sol_H1_d0.reshape([N_sample, len(t_pred)])
Sol_H_d0_mat = np.transpose(Sol_H2_d0) 
Sol_H_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_H_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_H_d0_mat, Weights)*0.5))   
Sol_R1_d0 = np.asarray(Sol_R_d0, order= 'F')
Sol_R2_d0 = Sol_R1_d0.reshape([N_sample, len(t_pred)])
Sol_R_d0_mat = np.transpose(Sol_R2_d0) 
Sol_R_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_R_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_R_d0_mat, Weights)*0.5))  
Sol_newI1_d0 = np.asarray(Sol_newI_d0, order= 'F')
Sol_newI2_d0 = Sol_newI1_d0.reshape([N_sample, len(t_pred)-1])
Sol_newI_d0_mat = np.transpose(Sol_newI2_d0) 
Sol_newI_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_newI_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newI_d0_mat, Weights)*0.5))   
Sol_newH1_d0 = np.asarray(Sol_newH_d0, order= 'F')
Sol_newH2_d0 = Sol_newH1_d0.reshape([N_sample, len(t_pred)-1])
Sol_newH_d0_mat = np.transpose(Sol_newH2_d0)  
Sol_newH_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_newH_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newH_d0_mat, Weights)*0.5))
Sol_newD1_d0 = np.asarray(Sol_newD_d0, order= 'F')
Sol_newD2_d0 = Sol_newD1_d0.reshape([N_sample, len(t_pred)-1])
Sol_newD_d0_mat = np.transpose(Sol_newD2_d0)  
Sol_newD_d0_std_V2 = np.sqrt(np.matmul(np.square(Sol_newD_d0_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newD_d0_mat, Weights)*0.5)) 

Sol_S1_d1 = np.asarray(Sol_S_d1, order= 'F')
Sol_S2_d1 = Sol_S1_d1.reshape([N_sample, len(t_pred)])
Sol_S_d1_mat = np.transpose(Sol_S2_d1) 
Sol_S_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_S_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_S_d1_mat, Weights)*0.5))  
Sol_I1_d1 = np.asarray(Sol_I_d1, order= 'F')
Sol_I2_d1 = Sol_I1_d1.reshape([N_sample, len(t_pred)])
Sol_I_d1_mat = np.transpose(Sol_I2_d1) 
Sol_I_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_I_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_I_d1_mat, Weights)*0.5))   
Sol_J1_d1 = np.asarray(Sol_J_d1, order= 'F')
Sol_J2_d1 = Sol_J1_d1.reshape([N_sample, len(t_pred)])
Sol_J_d1_mat = np.transpose(Sol_J2_d1) 
Sol_J_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_J_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_J_d1_mat, Weights)*0.5))   
Sol_D1_d1 = np.asarray(Sol_D_d1, order= 'F')
Sol_D2_d1 = Sol_D1_d1.reshape([N_sample, len(t_pred)])
Sol_D_d1_mat = np.transpose(Sol_D2_d1) 
Sol_D_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_D_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_D_d1_mat, Weights)*0.5))  
Sol_H1_d1 = np.asarray(Sol_H_d1, order= 'F')
Sol_H2_d1 = Sol_H1_d1.reshape([N_sample, len(t_pred)])
Sol_H_d1_mat = np.transpose(Sol_H2_d1) 
Sol_H_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_H_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_H_d1_mat, Weights)*0.5))   
Sol_R1_d1 = np.asarray(Sol_R_d1, order= 'F')
Sol_R2_d1 = Sol_R1_d1.reshape([N_sample, len(t_pred)])
Sol_R_d1_mat = np.transpose(Sol_R2_d1) 
Sol_R_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_R_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_R_d1_mat, Weights)*0.5))  
Sol_newI1_d1 = np.asarray(Sol_newI_d1, order= 'F')
Sol_newI2_d1 = Sol_newI1_d1.reshape([N_sample, len(t_pred)-1])
Sol_newI_d1_mat = np.transpose(Sol_newI2_d1) 
Sol_newI_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_newI_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newI_d1_mat, Weights)*0.5))    
Sol_newH1_d1 = np.asarray(Sol_newH_d1, order= 'F')
Sol_newH2_d1 = Sol_newH1_d1.reshape([N_sample, len(t_pred)-1])
Sol_newH_d1_mat = np.transpose(Sol_newH2_d1)    
Sol_newH_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_newH_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newH_d1_mat, Weights)*0.5)) 
Sol_newD1_d1 = np.asarray(Sol_newD_d1, order= 'F')
Sol_newD2_d1 = Sol_newD1_d1.reshape([N_sample, len(t_pred)-1])
Sol_newD_d1_mat = np.transpose(Sol_newD2_d1)   
Sol_newD_d1_std_V2 = np.sqrt(np.matmul(np.square(Sol_newD_d1_mat), Weights)*0.5-\
                         np.square(np.matmul(Sol_newD_d1_mat, Weights)*0.5))  
     
Sol_mean_V2 = odeint(ODEs_mean_2, U_init, t_pred.flatten(), args = (0,0))      
Sol_S_mean_V2 = Sol_mean_V2[:,0]
Sol_I_mean_V2 = Sol_mean_V2[:,1] 
Sol_J_mean_V2 = Sol_mean_V2[:,2] 
Sol_D_mean_V2 = Sol_mean_V2[:,3] 
Sol_H_mean_V2 = Sol_mean_V2[:,4] 
Sol_R_mean_V2 = Sol_mean_V2[:,5] 
Sol_sumI_mean_V2 = Sol_mean_V2[:,6] 
Sol_sumH_mean_V2 = Sol_mean_V2[:,7]     
Sol_newI_mean_V2 = np.diff(Sol_sumI_mean_V2)
Sol_newH_mean_V2 = np.diff(Sol_sumH_mean_V2)
Sol_newD_mean_V2 = np.diff(Sol_D_mean_V2)  
Diff_newI = I_new_PINN[-1]/Sol_newI_mean_V2[0] 
Diff_newH = H_new_PINN[-1]/Sol_newH_mean_V2[0] 
Diff_newD = D_new_PINN[-1]/Sol_newD_mean_V2[0]  
Sol_newI_mean_V2 = Sol_newI_mean_V2 * Diff_newI 
Sol_newH_mean_V2 = Sol_newH_mean_V2 * Diff_newH 
Sol_newD_mean_V2 = Sol_newD_mean_V2 * Diff_newD  

#%%
######################################################################
######################################################################
############################# Save the results ###############################
######################################################################
###################################################################### 
#%% 
#saver
current_directory = os.getcwd()
relative_path = '/Model7/Prediction-Results-'+dt_string+'/'
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    

np.savetxt(save_results_to + 'Sol_S_mean_V1.txt', Sol_S_mean_V1.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_I_mean_V1.txt', Sol_I_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_J_mean_V1.txt', Sol_J_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_D_mean_V1.txt', Sol_D_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_H_mean_V1.txt', Sol_H_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_R_mean_V1.txt', Sol_R_mean_V1.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_newI_mean_V1.txt', Sol_newI_mean_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newH_mean_V1.txt', Sol_newH_mean_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_newD_mean_V1.txt', Sol_newD_mean_V1.reshape((-1,1)))  

np.savetxt(save_results_to + 'Sol_S_d0_std_V1.txt', Sol_S_d0_std_V1.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_I_d0_std_V1.txt', Sol_I_d0_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_J_d0_std_V1.txt', Sol_J_d0_std_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_D_d0_std_V1.txt', Sol_D_d0_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_H_d0_std_V1.txt', Sol_H_d0_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_R_d0_std_V1.txt', Sol_R_d0_std_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newI_d0_std_V1.txt', Sol_newI_d0_std_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newH_d0_std_V1.txt', Sol_newH_d0_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_newD_d0_std_V1.txt', Sol_newD_d0_std_V1.reshape((-1,1))) 

np.savetxt(save_results_to + 'Sol_S_d1_std_V1.txt', Sol_S_d1_std_V1.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_I_d1_std_V1.txt', Sol_I_d1_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_J_d1_std_V1.txt', Sol_J_d1_std_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_D_d1_std_V1.txt', Sol_D_d1_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_H_d1_std_V1.txt', Sol_H_d1_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_R_d1_std_V1.txt', Sol_R_d1_std_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newI_d1_std_V1.txt', Sol_newI_d1_std_V1.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newH_d1_std_V1.txt', Sol_newH_d1_std_V1.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_newD_d1_std_V1.txt', Sol_newD_d1_std_V1.reshape((-1,1)))  

np.savetxt(save_results_to + 'Sol_S_mean_V2.txt', Sol_S_mean_V2.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_I_mean_V2.txt', Sol_I_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_J_mean_V2.txt', Sol_J_mean_V2.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_D_mean_V2.txt', Sol_D_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_H_mean_V2.txt', Sol_H_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_R_mean_V2.txt', Sol_R_mean_V2.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_newI_mean_V2.txt', Sol_newI_mean_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newH_mean_V2.txt', Sol_newH_mean_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_newD_mean_V2.txt', Sol_newD_mean_V2.reshape((-1,1)))  

np.savetxt(save_results_to + 'Sol_S_d0_std_V2.txt', Sol_S_d0_std_V2.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_I_d0_std_V2.txt', Sol_I_d0_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_J_d0_std_V2.txt', Sol_J_d0_std_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_D_d0_std_V2.txt', Sol_D_d0_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_H_d0_std_V2.txt', Sol_H_d0_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_R_d0_std_V2.txt', Sol_R_d0_std_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newI_d0_std_V2.txt', Sol_newI_d0_std_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newH_d0_std_V2.txt', Sol_newH_d0_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_newD_d0_std_V2.txt', Sol_newD_d0_std_V2.reshape((-1,1))) 

np.savetxt(save_results_to + 'Sol_S_d1_std_V2.txt', Sol_S_d1_std_V2.reshape((-1,1)))  
np.savetxt(save_results_to + 'Sol_I_d1_std_V2.txt', Sol_I_d1_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_J_d1_std_V2.txt', Sol_J_d1_std_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_D_d1_std_V2.txt', Sol_D_d1_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_H_d1_std_V2.txt', Sol_H_d1_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_R_d1_std_V2.txt', Sol_R_d1_std_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newI_d1_std_V2.txt', Sol_newI_d1_std_V2.reshape((-1,1))) 
np.savetxt(save_results_to + 'Sol_newH_d1_std_V2.txt', Sol_newH_d1_std_V2.reshape((-1,1)))
np.savetxt(save_results_to + 'Sol_newD_d1_std_V2.txt', Sol_newD_d1_std_V2.reshape((-1,1)))  


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


ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
# ax.legend(fontsize=35, ncol = 1, loc = 'best')
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
                      Sol_newI_mean_V1.flatten()-Sol_newI_d0_std_V1.flatten(), \
                      Sol_newI_mean_V1.flatten()+Sol_newI_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newI_mean_V1.flatten()-Sol_newI_d1_std_V1.flatten(), \
                      Sol_newI_mean_V1.flatten()+Sol_newI_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newI_mean_V2.flatten()-Sol_newI_d0_std_V2.flatten(), \
                      Sol_newI_mean_V2.flatten()+Sol_newI_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newI_mean_V2.flatten()-Sol_newI_d1_std_V2.flatten(), \
                      Sol_newI_mean_V2.flatten()+Sol_newI_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred[:-1], Sol_newI_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred[:-1], Sol_newI_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(date_total, I_new_star, 'ro', lw=4, markersize=8, label='Data-7davg')
        # ax.plot(data_mean[:-1], I_new_PINN, 'k-', lw=4, label='PINN-Training')   
        ax.plot(data_mean[1:], I_new_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred[:-1], Sol_newI_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred[:-1], Sol_newI_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('daily cases', fontsize = 40) 
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
                      Sol_newH_mean_V1.flatten()-Sol_newH_d0_std_V1.flatten(), \
                      Sol_newH_mean_V1.flatten()+Sol_newH_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newH_mean_V1.flatten()-Sol_newH_d1_std_V1.flatten(), \
                      Sol_newH_mean_V1.flatten()+Sol_newH_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newH_mean_V2.flatten()-Sol_newH_d0_std_V2.flatten(), \
                      Sol_newH_mean_V2.flatten()+Sol_newH_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newH_mean_V2.flatten()-Sol_newH_d1_std_V2.flatten(), \
                      Sol_newH_mean_V2.flatten()+Sol_newH_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred[:-1], Sol_newH_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred[:-1], Sol_newH_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(date_total, H_new_star, 'ro', lw=4, markersize=8, label='Data-7davg')
        # ax.plot(data_mean[:-1], H_new_PINN, 'k-', lw=4, label='PINN-Training')   
        ax.plot(data_mean[1:], H_new_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred[:-1], Sol_newH_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred[:-1], Sol_newH_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('daily hospitalized cases', fontsize = 40) 
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
                      Sol_newD_mean_V1.flatten()-Sol_newD_d0_std_V1.flatten(), \
                      Sol_newD_mean_V1.flatten()+Sol_newD_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newD_mean_V1.flatten()-Sol_newD_d1_std_V1.flatten(), \
                      Sol_newD_mean_V1.flatten()+Sol_newD_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newD_mean_V2.flatten()-Sol_newD_d0_std_V2.flatten(), \
                      Sol_newD_mean_V2.flatten()+Sol_newD_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred[:-1].flatten(), \
                      Sol_newD_mean_V2.flatten()-Sol_newD_d1_std_V2.flatten(), \
                      Sol_newD_mean_V2.flatten()+Sol_newD_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred[:-1], Sol_newD_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred[:-1], Sol_newD_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(date_total, D_new_star, 'ro', lw=4, markersize=8, label='Data-7davg')
        # ax.plot(data_mean[:-1], D_new_PINN, 'k-', lw=4, label='PINN-Training')   
        ax.plot(data_mean[1:], D_new_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred[:-1], Sol_newD_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred[:-1], Sol_newD_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('daily death cases', fontsize = 40) 
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
                      Sol_S_mean_V1.flatten()-Sol_S_d0_std_V1.flatten(), \
                      Sol_S_mean_V1.flatten()+Sol_S_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_S_mean_V1.flatten()-Sol_S_d1_std_V1.flatten(), \
                      Sol_S_mean_V1.flatten()+Sol_S_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_S_mean_V2.flatten()-Sol_S_d0_std_V2.flatten(), \
                      Sol_S_mean_V2.flatten()+Sol_S_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_S_mean_V2.flatten()-Sol_S_d1_std_V2.flatten(), \
                      Sol_S_mean_V2.flatten()+Sol_S_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, Sol_S_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, Sol_S_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, S_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, Sol_S_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, Sol_S_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbb{S}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Suspectious.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Suspectious.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Suspectious_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Suspectious_zoom.png', dpi=300)  

#%% 
# #Current Exposed
# for i in [1,2]:
#     fig, ax = plt.subplots() 
#     plt.fill_between(data_pred.flatten(), \
#                       Sol_E_mean_V1.flatten()-Sol_E_d0_std_V1.flatten(), \
#                       Sol_E_mean_V1.flatten()+Sol_E_d0_std_V1.flatten(), \
#                       facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
#     plt.fill_between(data_pred.flatten(), \
#                       Sol_E_mean_V1.flatten()-Sol_E_d1_std_V1.flatten(), \
#                       Sol_E_mean_V1.flatten()+Sol_E_d1_std_V1.flatten(), \
#                       facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
#     plt.fill_between(data_pred.flatten(), \
#                       Sol_E_mean_V2.flatten()-Sol_E_d0_std_V2.flatten(), \
#                       Sol_E_mean_V2.flatten()+Sol_E_d0_std_V2.flatten(), \
#                       facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
#     plt.fill_between(data_pred.flatten(), \
#                       Sol_E_mean_V2.flatten()-Sol_E_d1_std_V2.flatten(), \
#                       Sol_E_mean_V2.flatten()+Sol_E_d1_std_V2.flatten(), \
#                       facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
#     if i==1:
#         ax.plot(data_pred, Sol_E_mean_V1, 'm--', lw=4, label='Prediction-mean')    
#         ax.plot(data_pred, Sol_E_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
#         ax.plot(data_mean, E_PINN, 'k-', lw=4, label='PINN-Training')   

#     if i==2:
#         ax.plot(data_pred, Sol_E_mean_V1, 'm--', lw=7, label='Prediction-mean')    
#         ax.plot(data_pred, Sol_E_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
#     # ax.set_xlim(200,300)
#     # ax.set_ylim(0-0.5,6000+0.5)
#     ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
#     ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
#     plt.xticks(rotation=30)
#     ax.legend(fontsize=35, ncol = 1, loc = 'best')
#     ax.tick_params(axis='x', labelsize = 40)
#     ax.tick_params(axis='y', labelsize = 40)
#     # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
#     plt.rc('font', size=40)
#     ax.grid(True)
#     # ax.set_xlabel('Date', fontsize = font) 
#     ax.set_ylabel('$\mathbb{E}$', fontsize = 40) 
#     fig.set_size_inches(w=25, h=12.5) 
#     if i==1:
#         plt.savefig(save_results_to + 'Current_Exposed.pdf', dpi=300) 
#         plt.savefig(save_results_to + 'Current_Exposed.png', dpi=300)  
#     if i==2:
#         plt.savefig(save_results_to + 'Current_Exposed_zoom.pdf', dpi=300) 
#         plt.savefig(save_results_to + 'Current_Exposed_zoom.png', dpi=300)  

#%% 
#Current infectious
for i in [1,2]:
    fig, ax = plt.subplots() 
    plt.fill_between(data_pred.flatten(), \
                      Sol_I_mean_V1.flatten()-Sol_I_d0_std_V1.flatten(), \
                      Sol_I_mean_V1.flatten()+Sol_I_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_I_mean_V1.flatten()-Sol_I_d1_std_V1.flatten(), \
                      Sol_I_mean_V1.flatten()+Sol_I_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_I_mean_V2.flatten()-Sol_I_d0_std_V2.flatten(), \
                      Sol_I_mean_V2.flatten()+Sol_I_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_I_mean_V2.flatten()-Sol_I_d1_std_V2.flatten(), \
                      Sol_I_mean_V2.flatten()+Sol_I_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, Sol_I_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, Sol_I_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, I_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, Sol_I_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, Sol_I_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbb{I}$', fontsize = 40) 
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
                      Sol_J_mean_V1.flatten()-Sol_J_d0_std_V1.flatten(), \
                      Sol_J_mean_V1.flatten()+Sol_J_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_J_mean_V1.flatten()-Sol_J_d1_std_V1.flatten(), \
                      Sol_J_mean_V1.flatten()+Sol_J_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_J_mean_V2.flatten()-Sol_J_d0_std_V2.flatten(), \
                      Sol_J_mean_V2.flatten()+Sol_J_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_J_mean_V2.flatten()-Sol_J_d1_std_V2.flatten(), \
                      Sol_J_mean_V2.flatten()+Sol_J_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, Sol_J_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, Sol_J_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, J_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, Sol_J_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, Sol_J_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbb{J}$', fontsize = 40) 
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
                      Sol_D_mean_V1.flatten()-Sol_D_d0_std_V1.flatten(), \
                      Sol_D_mean_V1.flatten()+Sol_D_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_D_mean_V1.flatten()-Sol_D_d1_std_V1.flatten(), \
                      Sol_D_mean_V1.flatten()+Sol_D_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_D_mean_V2.flatten()-Sol_D_d0_std_V2.flatten(), \
                      Sol_D_mean_V2.flatten()+Sol_D_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_D_mean_V2.flatten()-Sol_D_d1_std_V2.flatten(), \
                      Sol_D_mean_V2.flatten()+Sol_D_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, Sol_D_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, Sol_D_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, D_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, Sol_D_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, Sol_D_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbb{D}$', fontsize = 40) 
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
                      Sol_H_mean_V1.flatten()-Sol_H_d0_std_V1.flatten(), \
                      Sol_H_mean_V1.flatten()+Sol_H_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_H_mean_V1.flatten()-Sol_H_d1_std_V1.flatten(), \
                      Sol_H_mean_V1.flatten()+Sol_H_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_H_mean_V2.flatten()-Sol_H_d0_std_V2.flatten(), \
                      Sol_H_mean_V2.flatten()+Sol_H_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_H_mean_V2.flatten()-Sol_H_d1_std_V2.flatten(), \
                      Sol_H_mean_V2.flatten()+Sol_H_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, Sol_H_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, Sol_H_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        # ax.plot(data_mean, H_star, 'ro', lw=4, markersize=8, label='Data-7davg')   
        ax.plot(data_mean, H_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, Sol_H_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, Sol_H_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbb{H}$', fontsize = 40) 
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
                      Sol_R_mean_V1.flatten()-Sol_R_d0_std_V1.flatten(), \
                      Sol_R_mean_V1.flatten()+Sol_R_d0_std_V1.flatten(), \
                      facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_R_mean_V1.flatten()-Sol_R_d1_std_V1.flatten(), \
                      Sol_R_mean_V1.flatten()+Sol_R_d1_std_V1.flatten(), \
                      facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_R_mean_V2.flatten()-Sol_R_d0_std_V2.flatten(), \
                      Sol_R_mean_V2.flatten()+Sol_R_d0_std_V2.flatten(), \
                      facecolor=(0.6,0.2,0.5,0.3), interpolate=True)#, label='Prediction-std-(10%)')  
    plt.fill_between(data_pred.flatten(), \
                      Sol_R_mean_V2.flatten()-Sol_R_d1_std_V2.flatten(), \
                      Sol_R_mean_V2.flatten()+Sol_R_d1_std_V2.flatten(), \
                      facecolor=(0.6,0.5,0.8,0.3), interpolate=True)#, label='Prediction-std-(20%)')   
    
    
    if i==1:
        ax.plot(data_pred, Sol_R_mean_V1, 'm--', lw=4, label='Prediction-mean')    
        ax.plot(data_pred, Sol_R_mean_V2, 'g--', lw=4, label='Prediction-mean (higher rate vacc.)')    
        ax.plot(data_mean, R_PINN, 'k-', lw=4, label='PINN-Training')   

    if i==2:
        ax.plot(data_pred, Sol_R_mean_V1, 'm--', lw=7, label='Prediction-mean')    
        ax.plot(data_pred, Sol_R_mean_V2, 'g--', lw=7, label='Prediction-mean (higher rate vacc.)')    

    
    # ax.set_xlim(200,300)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervals[i]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    plt.rc('font', size=40)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$\mathbb{R}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    if i==1:
        plt.savefig(save_results_to + 'Current_Recovered.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Recovered.png', dpi=300)  
    if i==2:
        plt.savefig(save_results_to + 'Current_Recovered_zoom.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Current_Recovered_zoom.png', dpi=300)  

