# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:13:45 2020 

@author: Min Cai 

"""


import sys
sys.path.insert(0, '../../Utilities/')
 
import pandas
# import tensorflow as tf
import matplotlib.dates as mdates
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io 
import time  
start_time = time.time()


# np.random.seed(1234)
# tf.set_random_seed(1234)
# # tf.random.set_seed(1234)

       
#%%
from datetime import datetime
now = datetime.now()
# dt_string = now.strftime("%m-%d")
dt_string =  '03-13'
date_total = np.arange('2020-03-06', '2021-03-11', dtype='datetime64[D]')[:,None] 

#%%
if __name__=="__main__":   
    
    #Scaling 
    sf = 1e-4 
    
    #Load data
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

#%%
    ######################################################################
    ############################# Legends ###############################
    ######################################################################
#%% 
    #Legends
    fig, ax = plt.subplots()   
    ax.plot(date_total, I_new_star*0.0, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, I_new_star*0.0, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, I_new_star*0.0, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
    # ax.plot(date_total, I_new_star*0.0, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)') 
    plt.ylim(1, 2)
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    # # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%y'))
    # ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    # plt.xticks(rotation=20)
    ax.legend(fontsize=120, ncol = 3, loc = 'center',  shadow=False)
    plt.axis('off')
    # ax.tick_params(axis='x', labelsize = 50)
    # ax.tick_params(axis='y', labelsize = 60)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    # plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font)
    # ax.set_ylabel('Current Susceptible ($S$)', fontsize = font) 
    # ax.set_ylabel(r'$\beta_{I}$', fontsize = 75) 
    fig.set_size_inches(w=80, h=10) 
    plt.savefig('Compare/Legends.pdf', dpi=300) 
    

#%%
    ######################################################################
    ############################# Parameters ###############################
    ###################################################################### 
#%%
    BetaI_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/BetaI_pred_mean.txt')
    BetaI_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/BetaI_pred_mean.txt')
    BetaI_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/BetaI_pred_mean.txt') 
    BetaI_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/BetaI_pred_mean.txt')
    
    Rc_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/Rc_pred_mean.txt')
    Rc_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/Rc_pred_mean.txt')
    Rc_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/Rc_pred_mean.txt')  
    Rc_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/Rc_pred_mean.txt')

    p_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/p_pred_mean.txt')
    p_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/p_pred_mean.txt')
    p_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/p_pred_mean.txt') 
    p_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/p_pred_mean.txt')
    
    q_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/q_pred_mean.txt') 
    q_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/q_pred_mean.txt')
    q_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/q_pred_mean.txt') 
    q_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/q_pred_mean.txt') 

#%% 
    #Beta Curve
    fig, ax = plt.subplots()   
    ax.plot(date_total, BetaI_SEIR, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, BetaI_SEPIR, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, BetaI_SEPIQR, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
    ax.plot(date_total, BetaI_Delay, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
    
    # ax.set_xlim('2020-03-01','2021-02-15')
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    plt.xticks(rotation=30) 
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.set_ylabel(r'$\beta_{I}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)  
    plt.savefig('Compare/BetaI_compare.pdf', dpi=300) 
    
#%%    
    #Rc Curve 
    fig, ax = plt.subplots()   
    ax.plot(date_total, Rc_SEIR, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, Rc_SEPIR, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, Rc_SEPIQR, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
    ax.plot(date_total, Rc_Delay, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    plt.xticks(rotation=30) 
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.set_ylabel('$R_{c}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    plt.savefig('Compare/Rc_compare.pdf', dpi=300)

#%%
    #p curve  
    fig, ax = plt.subplots()   
    ax.plot(date_total, p_SEIR, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, p_SEPIR, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, p_SEPIQR, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
    ax.plot(date_total, p_Delay, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)') 
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    plt.xticks(rotation=30) 
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.set_ylabel('$p$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    plt.savefig('Compare/p_compare.pdf', dpi=300)  

#%%
    #q curve  
    fig, ax = plt.subplots()   
    ax.plot(date_total, q_SEIR, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, q_SEPIR, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, q_SEPIQR, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
    ax.plot(date_total, q_Delay, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    plt.xticks(rotation=30) 
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.set_ylabel('$q$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    plt.savefig('Compare/q_compare.pdf', dpi=300)  
    
#%%
    ######################################################################
    ############################# Fitting ###############################
    ###################################################################### 
#%%
    I_new_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I_new_pred_mean.txt')
    I_new_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/I_new_pred_mean.txt')
    I_new_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/I_new_pred_mean.txt')  
    I_new_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/I_new_pred_mean.txt')
    
    H_new_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/H_new_pred_mean.txt')
    H_new_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/H_new_pred_mean.txt')
    H_new_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/H_new_pred_mean.txt')  
    H_new_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/H_new_pred_mean.txt')
    
    D_new_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/D_new_pred_mean.txt')
    D_new_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/D_new_pred_mean.txt')
    D_new_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/D_new_pred_mean.txt') 
    D_new_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/D_new_pred_mean.txt')
    
    I_sum_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I_sum_pred_mean.txt')
    I_sum_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/I_sum_pred_mean.txt')
    I_sum_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/I_sum_pred_mean.txt')  
    I_sum_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/I_sum_pred_mean.txt')
    
    H_sum_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/H_sum_pred_mean.txt')
    H_sum_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/H_sum_pred_mean.txt')
    H_sum_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/H_sum_pred_mean.txt') 
    H_sum_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/H_sum_pred_mean.txt')
    
    D_sum_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/D_sum_pred_mean.txt')
    D_sum_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/D_sum_pred_mean.txt')
    D_sum_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/D_sum_pred_mean.txt') 
    D_sum_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/D_sum_pred_mean.txt')
    
#%%     
    #New infectious 
    fig, ax = plt.subplots()  
    ax.plot(date_total, I_new_star, 'ro', lw=3, markersize=10, label='Data-7davg')
    ax.plot(date_total[1:], I_new_SEIR/sf, 'k-', lw=4, label='PINN') 
    # ax.plot(date_total, I_new_SEIR/sf, 'k-', lw=2, label='model ($\mathbb{I}_{1}$)') 
    # ax.plot(date_total, I_new_SEPIR/sf, 'k--', lw=2, label='model ($\mathbb{I}_{2}$)') 
    # ax.plot(date_total, I_new_SEPIQR/sf, 'k:', lw=2, label='model ($\mathbb{I}_{3}$)')   
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.set_ylabel('Daily infectious cases', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/New_infectious_compare.pdf', dpi=300) 
    
#%%    
    #New hospitalized 
    fig, ax = plt.subplots()  
    ax.plot(date_total, H_new_star, 'ro', lw=3, markersize=10, label='Data-7davg')
    ax.plot(date_total[1:], H_new_SEIR/sf, 'k-', lw=4, label='PINN') 
    # ax.plot(date_total, H_new_SEIR/sf, 'k-', lw=4, label='model ($\mathbb{I}_{1}$)') 
    # ax.plot(date_total, H_new_SEPIR/sf, 'r-', lw=4, label='model ($\mathbb{I}_{2}$)') 
    # ax.plot(date_total, H_new_SEPIQR/sf, 'b-', lw=4, label='model ($\mathbb{I}_{3}$)')   
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('Daily hospitalized cases', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/New_hospitalized_compare.pdf', dpi=300) 
    
#%%    
    #New death 
    fig, ax = plt.subplots()  
    ax.plot(date_total, D_new_star, 'ro', lw=3, markersize=10, label='Data-7davg')
    ax.plot(date_total[1:], D_new_SEIR/sf, 'k-', lw=4, label='PINN')  
    # ax.plot(date_total, D_new_SEIR/sf, 'k-', lw=4, label='model ($\mathbb{I}_{1}$)') 
    # ax.plot(date_total, D_new_SEPIR/sf, 'r-', lw=4, label='model ($\mathbb{I}_{2}$)') 
    # ax.plot(date_total, D_new_SEPIQR/sf, 'b-', lw=4, label='model ($\mathbb{I}_{3}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('Daily death cases', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/New_death_compare.pdf', dpi=300) 
    
#%%
    #Cumulative infectious 
    fig, ax = plt.subplots()  
    ax.plot(date_total, I_sum_star, 'ro', lw=3, markersize=10, label='Data-7davg')
    ax.plot(date_total, I_sum_SEIR/sf, 'k-', lw=4, label='PINN')  
    # ax.plot(date_total, I_sum_SEIR/sf, 'k-', lw=4, label='model ($\mathbb{I}_{1}$)') 
    # ax.plot(date_total, I_sum_SEPIR/sf, 'r-', lw=4, label='model ($\mathbb{I}_{2}$)') 
    # ax.plot(date_total, I_sum_SEPIQR/sf, 'b-', lw=4, label='model ($\mathbb{I}_{3}$)')   
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('Cumulative infectious cases', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Cumulative_infectious_compare.pdf', dpi=300) 
    
#%%    
    #Cumulative hospitalized 
    fig, ax = plt.subplots()  
    ax.plot(date_total, H_sum_star, 'ro', lw=3, markersize=10, label='Data-7davg')
    ax.plot(date_total, H_sum_SEIR/sf, 'k-', lw=4, label='PINN')  
    # ax.plot(date_total, H_sum_SEIR/sf, 'k-', lw=4, label='model ($\mathbb{I}_{1}$)') 
    # ax.plot(date_total, H_sum_SEPIR/sf, 'r-', lw=4, label='model ($\mathbb{I}_{2}$)') 
    # ax.plot(date_total, H_sum_SEPIQR/sf, 'b-', lw=4, label='model ($\mathbb{I}_{3}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('Cumulative hospitalized cases', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5) 
    plt.savefig('Compare/Cumulative_hospitalized_compare.pdf', dpi=300) 

#%%    
    #Cumulative death 
    fig, ax = plt.subplots()  
    ax.plot(date_total, D_sum_star, 'ro', lw=3, markersize=10, label='Data-7davg')
    ax.plot(date_total, D_sum_SEIR/sf, 'k-', lw=4, label='PINN')   
    # ax.plot(date_total, D_sum_SEIR/sf, 'k-', lw=4, label='model ($\mathbb{I}_{1}$)') 
    # ax.plot(date_total, D_sum_SEPIR/sf, 'r-', lw=4, label='model ($\mathbb{I}_{2}$)') 
    # ax.plot(date_total, D_sum_SEPIQR/sf, 'b-', lw=4, label='model ($\mathbb{I}_{3}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('Cumulative death cases', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Cumulative_death_compare.pdf', dpi=300)
    
#%%
    ######################################################################
    ########################## Current Values ############################
    ###################################################################### 
#%%   
    #Current Susceptible  
    S_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/S_pred_mean.txt')
    S_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/S_pred_mean.txt')
    S_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/S_pred_mean.txt') 
    S_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/S_pred_mean.txt') 
 
    fig, ax = plt.subplots()   
    ax.plot(date_total, S_SEIR/sf, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, S_SEPIR/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, S_SEPIQR/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
    ax.plot(date_total, S_Delay/sf, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{S}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Susceptible_compare.pdf', dpi=300)  
    
#%%   
    #Current Exposed   
    E_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/E_pred_mean.txt')
    E_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/E_pred_mean.txt')
    E_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/E_pred_mean.txt') 
 
    fig, ax = plt.subplots()   
    ax.plot(date_total, E_SEIR/sf, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, E_SEPIR/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, E_SEPIQR/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{E}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Exposed_compare.pdf', dpi=300)  
    
#%%   
    #Current Infections  
    I_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I_pred_mean.txt')
    I_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/I_pred_mean.txt')
    I_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/I_pred_mean.txt') 
    I_Delay= np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/I_pred_mean.txt') 
 
    fig, ax = plt.subplots()   
    ax.plot(date_total, I_SEIR/sf, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, I_SEPIR/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, I_SEPIQR/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)')    
    ax.plot(date_total, I_Delay/sf, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{I}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Infections_compare.pdf', dpi=300)  
    
#%%   
    #Current Asymptomatic  
    J_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/J_pred_mean.txt')
    J_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/J_pred_mean.txt')
    J_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/J_pred_mean.txt') 
    J_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/J_pred_mean.txt') 
 
    fig, ax = plt.subplots()   
    ax.plot(date_total, J_SEIR/sf, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, J_SEPIR/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, J_SEPIQR/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)')     
    ax.plot(date_total, J_Delay/sf, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{J}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Asymptomatic_compare.pdf', dpi=300)  

#%%   
    #Current Hospitalizations 
    H_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/H_pred_mean.txt')
    H_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/H_pred_mean.txt')
    H_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/H_pred_mean.txt') 
    H_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/H_pred_mean.txt') 
 
    fig, ax = plt.subplots()   
    ax.plot(date_total, H_SEIR/sf, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, H_SEPIR/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, H_SEPIQR/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)')    
    ax.plot(date_total, H_Delay/sf, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{H}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_hospitalized_compare.pdf', dpi=300)  

#%%   
    #Current Death  
    D_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/D_pred_mean.txt')
    D_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/D_pred_mean.txt')
    D_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/D_pred_mean.txt') 
    D_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/D_sum_pred_mean.txt') 
 
    fig, ax = plt.subplots()   
    ax.plot(date_total, D_SEIR/sf, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, D_SEPIR/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, D_SEPIQR/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)')      
    ax.plot(date_total, D_Delay/sf, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{D}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Death_compare.pdf', dpi=300)  
    
#%%   
    #Current Recovered  
    R_SEIR = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/R_pred_mean.txt')
    R_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/R_pred_mean.txt')
    R_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/R_pred_mean.txt') 
    R_Delay = np.loadtxt('Model7/Train-Results-'+dt_string+'-Average/R_pred_mean.txt') 
 
    fig, ax = plt.subplots()   
    ax.plot(date_total, R_SEIR/sf, 'k-', lw=5, label='model ($\mathbb{I}_{1}$)') 
    ax.plot(date_total, R_SEPIR/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, R_SEPIQR/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)')      
    ax.plot(date_total, R_Delay/sf, 'c--', lw=5, label='model ($\mathbb{T}_{1}$)')  
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{R}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Recovered_compare.pdf', dpi=300)  
    
#%%   
    #Current Pre-symptomatic  
    PreS_SEPIR = np.loadtxt('Model2/Train-Results-'+dt_string+'-Average/PreS_pred_mean.txt')
    PreS_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/PreS_pred_mean.txt')
 
    fig, ax = plt.subplots()    
    ax.plot(date_total, PreS_SEPIR[0:len(date_total)]/sf, 'b-.', lw=5, label='model ($\mathbb{I}_{2}$)') 
    ax.plot(date_total, PreS_SEPIQR[0:len(date_total)]/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{P}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Pre-symptomatic_compare.pdf', dpi=300)  
    
#%%    
    #Current Quarantine  
    Qua_SEPIQR = np.loadtxt('Model3/Train-Results-'+dt_string+'-Average/Qua_pred_mean.txt')
 
    fig, ax = plt.subplots()    
    ax.plot(date_total, Qua_SEPIQR[0:len(date_total)]/sf, 'r:', lw=8, label='model ($\mathbb{I}_{3}$)') 
     
    # ax.set_xlim('2020-03-01','2021-02-15')
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30) 
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=40)
    # ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.set_ylabel('$\mathbf{Q}$', fontsize = 40) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig('Compare/Current_Quarantine_compare.pdf', dpi=300) 
