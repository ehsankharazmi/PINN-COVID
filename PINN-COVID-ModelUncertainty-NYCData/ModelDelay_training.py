# -*- coding: utf-8 -*-
""" 
@Title:
    Identifiability and predictability of
    integer- and fractional-order epidemiological models
    using physics-informed neural networks
@author: 
    Ehsan Kharazmi & Min Cai $ Xiaoning Zheng
    Division of Applied Mathematics
    Brown University
    ehsan_kharazmi@brown.edu
Created on 2020 
"""


import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas
import math 
import tensorflow as tf
import numpy as np
from numpy import *
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
start_time = time.time()


# np.random.seed(1234)
# tf.set_random_seed(1234)
# tf.random.set_seed(1234)

#%%
a=19 #The first 20 days have no delay 
# b=30
# T=271
class PhysicsInformedNN:
    #Initialize the class
    def __init__(self, t_train, I_new_train, D_new_train, H_new_train, 
                 I_sum_train, D_sum_train, H_sum_train, U0, t_f, lb, ub, N, 
                 layers, layers_beta, layers_p, layers_q, sf): 
        
        self.N = N
        self.sf = sf
        
        #Data for training  
        self.t_train = t_train  
        self.I_new_train = I_new_train 
        self.D_new_train = D_new_train  
        self.H_new_train = H_new_train
        self.I_sum_train = I_sum_train 
        self.D_sum_train = D_sum_train  
        self.H_sum_train = H_sum_train 
        self.S0 = U0[0] 
        self.E0 = U0[1] 
        self.I0 = U0[2]
        self.J0 = U0[3] 
        self.D0 = U0[4]  
        self.H0 = U0[5]  
        self.R0 = U0[6]  
        self.t_f = t_f  
        
        #Time division s
        self.M = len(t_f)-1
        self.tau = t_f[1]-t_f[0] 

        #Bounds 
        self.lb = lb
        self.ub = ub

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  
        self.weights_beta, self.biases_beta = self.initialize_NN(layers_beta) 
        self.weights_p, self.biases_p = self.initialize_NN(layers_p) 
        self.weights_q, self.biases_q = self.initialize_NN(layers_q)

       #Fixed parameters
        self.N = N
        self.delta = tf.Variable(0.6,dtype=tf.float64,trainable=False)
        self.gamma = tf.Variable(1.0/6.0,dtype=tf.float64,trainable=False)
        self.gammaA = tf.Variable(1.0/6.0,dtype=tf.float64,trainable=False)  
        self.alpha = tf.Variable(1.0/5.2,dtype=tf.float64,trainable=False)
        self.alpha1 = tf.Variable(1.0/2.9,dtype=tf.float64,trainable=False)
        self.alpha2 = tf.Variable(1.0/2.3,dtype=tf.float64,trainable=False) 
        self.eps1 = tf.Variable(0.75,dtype=tf.float64,trainable=False) 
        self.eps2 = tf.Variable(0.0,dtype=tf.float64,trainable=False) 
        self.phiD = tf.Variable(1.0/15.0,dtype=tf.float64,trainable=False) 
        self.phiR = tf.Variable(1.0/7.5,dtype=tf.float64,trainable=False)   
        self.d1 = tf.Variable(0,dtype=tf.float64,trainable=False)  
        self.d2 = 3.0+(10.0-3.0)*tf.sigmoid(tf.Variable(0,dtype=tf.float64,trainable=True))
        self.d3 = tf.Variable(0,dtype=tf.float64,trainable=False) 
        
       # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.saver = tf.train.Saver()

        # placeholders for inputs    
        self.t_u = tf.placeholder(tf.float64, shape=[None, self.t_train.shape[1]])  
        self.I_new_u = tf.placeholder(tf.float64, shape=[None, self.I_new_train.shape[1]]) 
        self.D_new_u = tf.placeholder(tf.float64, shape=[None, self.D_new_train.shape[1]])  
        self.H_new_u = tf.placeholder(tf.float64, shape=[None, self.H_new_train.shape[1]]) 
        self.I_sum_u = tf.placeholder(tf.float64, shape=[None, self.I_sum_train.shape[1]]) 
        self.D_sum_u = tf.placeholder(tf.float64, shape=[None, self.D_sum_train.shape[1]])  
        self.H_sum_u = tf.placeholder(tf.float64, shape=[None, self.H_sum_train.shape[1]]) 
        self.S0_u = tf.placeholder(tf.float64, shape=[None, self.S0.shape[1]]) 
        self.E0_u = tf.placeholder(tf.float64, shape=[None, self.E0.shape[1]]) 
        self.I0_u = tf.placeholder(tf.float64, shape=[None, self.I0.shape[1]]) 
        self.J0_u = tf.placeholder(tf.float64, shape=[None, self.J0.shape[1]]) 
        self.D0_u = tf.placeholder(tf.float64, shape=[None, self.D0.shape[1]]) 
        self.H0_u = tf.placeholder(tf.float64, shape=[None, self.H0.shape[1]]) 
        self.R0_u = tf.placeholder(tf.float64, shape=[None, self.R0.shape[1]]) 
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])   

        # physics informed neural networks
        self.S_pred, self.I_pred, self.J_pred, self.H_pred, self.D_pred, self.R_pred, self.I_sum_pred, self.H_sum_pred = self.net_u(self.t_u)  
        self.D_sum_pred = self.D_pred
        
        self.BetaI_pred = self.net_BetaI(self.t_u)          
        self.p_pred = self.net_p(self.t_u)  
        self.q_pred = self.net_q(self.t_u)  
        
        self.I_new_pred = self.I_sum_pred[1:,:]-self.I_sum_pred[0:-1,:]
        self.H_new_pred = self.H_sum_pred[1:,:]-self.H_sum_pred[0:-1,:]
        self.D_new_pred = self.D_sum_pred[1:,:]-self.D_sum_pred[0:-1,:]
        
        self.S0_pred = self.S_pred[0] 
        self.I0_pred = self.I_pred[0]
        self.J0_pred = self.J_pred[0]
        self.D0_pred = self.D_pred[0]
        self.H0_pred = self.H_pred[0]  
        self.R0_pred = self.R_pred[0]   
        
        self.I_f, self.J_f, self.H_f, self.D_f, self.R_f, self.I_sum_f, self.H_sum_f = self.net_f(self.t_tf)
        
        # loss 
        self.lossU0 = tf.reduce_mean(tf.square(self.I0_u - self.I0_pred)) + \
            tf.reduce_mean(tf.square(self.J0_u - self.J0_pred)) + \
            tf.reduce_mean(tf.square(self.D0_u - self.D0_pred)) + \
            tf.reduce_mean(tf.square(self.H0_u - self.H0_pred)) + \
            tf.reduce_mean(tf.square(self.R0_u - self.R0_pred))
            # tf.reduce_mean(tf.square(self.S0_u - self.S0_pred))  
        
        # self.lossU = tf.reduce_mean(tf.square(self.I_new_u[:-1,:] - self.I_new_pred)) + \
        #     tf.reduce_mean(tf.square(self.D_new_u[:-1,:] - self.D_new_pred)) + \
        #     tf.reduce_mean(tf.square(self.H_new_u[:-1,:] - self.H_new_pred)) + \
        #     tf.reduce_mean(tf.square(self.I_sum_u - self.I_sum_pred)) + \
        #     tf.reduce_mean(tf.square(self.D_sum_u - self.D_sum_pred)) + \
        #     tf.reduce_mean(tf.square(self.H_sum_u - self.H_sum_pred)) 
        
        self.lossU = 120*tf.reduce_mean(tf.square(self.I_new_u[:-1,:] - self.I_new_pred)) + \
            10*120*tf.reduce_mean(tf.square(self.D_new_u[:-1,:] - self.D_new_pred)) + \
            3*120*tf.reduce_mean(tf.square(self.H_new_u[:-1,:] - self.H_new_pred)) + \
            tf.reduce_mean(tf.square(self.I_sum_u - self.I_sum_pred)) + \
            10*tf.reduce_mean(tf.square(self.D_sum_u - self.D_sum_pred)) + \
            3*tf.reduce_mean(tf.square(self.H_sum_u - self.H_sum_pred))
        
        self.lossF = tf.reduce_mean(tf.square(self.I_f)) + \
            tf.reduce_mean(tf.square(self.J_f)) + tf.reduce_mean(tf.square(self.H_f)) + \
            tf.reduce_mean(tf.square(self.D_f)) + tf.reduce_mean(tf.square(self.R_f)) + \
            tf.reduce_mean(tf.square(self.I_sum_f)) + tf.reduce_mean(tf.square(self.H_sum_f)) 

        self.loss = self.lossU0 + self.lossU + self.lossF  


        #Optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init) 
        
    #Initialize the nueral network
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]]) 
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)  
            weights.append(W) 
            biases.append(b) 
        return weights, biases

    #generating weights
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    #Architecture of the neural network 
    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(t-self.lb)/(self.ub-self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y    

    def net_u(self, t):
        SIJDHR = self.neural_net(t, self.weights, self.biases)
        # SIJDHR = SIJDHR**2
        I = SIJDHR[:,0:1] 
        J = SIJDHR[:,1:2]
        H = SIJDHR[:,2:3]
        D = SIJDHR[:,3:4]
        R = SIJDHR[:,4:5]
        I_sum = SIJDHR[:,5:6]   
        H_sum = SIJDHR[:,6:7]  
        S = self.N - I - J - H - D - R
        return S, I, J, H, D, R, I_sum, H_sum    
    
    def net_BetaI(self,t):
        BetaI = self.neural_net(t, self.weights_beta, self.biases_beta) 
        bound_b = [tf.constant(0.05, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64)] 
        return bound_b[0]+(bound_b[1]-bound_b[0])*tf.sigmoid(BetaI)
    
    def net_p(self, t):
        p = self.neural_net(t, self.weights_p, self.biases_p) 
        return tf.sigmoid(p)
    
    def net_q(self,t):
        q = self.neural_net(t, self.weights_q, self.biases_q)
        return 0.15+(0.6-0.15)*tf.sigmoid(q)
        # return tf.sigmoid(q) 
    
    def net_f(self, t):
        #load fixed parameters
        delta = self.delta 
        eps1 = self.eps1 
        eps2 = self.eps2 
        alpha = self.alpha 
        gamma = self.gamma 
        gammaA = self.gammaA 
        phiD = self.phiD
        phiR = self.phiR 
        d1=self.d1
        d2=self.d2
        d3=self.d3
        
        #load time-dependent parameters
        betaI = self.net_BetaI(t)
        betaId1 = self.net_BetaI(t-d1) 
        betaId2 = self.net_BetaI(t-d2)
        betaId3 = self.net_BetaI(t-d3)
        p = self.net_p(t)
        q = self.net_q(t)   
        
        #Obtain S,E,I,J,D,H,R from Neural network
        S, I, J, H, D, R, I_sum, H_sum = self.net_u(t)
        
        Sd1, Id1, Jd1, Hd1, Dd1, Rd1, I_sumd1, H_sumd1 = self.net_u(t-d1)
        
        Sd2, Id2, Jd2, Hd2, Dd2, Rd2, I_sumd2, H_sumd2 = self.net_u(t-d2) 
        
        Sd3, Id3, Jd3, Hd3, Dd3, Rd3, I_sumd3, H_sumd3 = self.net_u(t-d3)
        
        #Time derivatives  
        I_t = tf.gradients(I, t)[0]
        J_t = tf.gradients(J, t)[0]
        D_t = tf.gradients(D, t)[0]
        H_t = tf.gradients(H, t)[0]
        R_t = tf.gradients(R, t)[0] 
        I_sum_t = tf.gradients(I_sum, t)[0]
        H_sum_t = tf.gradients(H_sum, t)[0]
        
        #tf.where(condition > 0.5, a, b)
        delay=tf.where(t<a,(betaI*(I+eps1*J+eps2*H)/self.N)*S,(betaId2*(Id2+eps1*Jd2+eps2*Hd2)/self.N)*Sd2)
        
        #Residuals  
        VCn = tf.where(t<300*tf.ones(tf.shape(t), dtype = tf.float64), \
                       tf.zeros(tf.shape(t), dtype = tf.float64), \
                           tf.where(t<350*tf.ones(tf.shape(t), dtype = tf.float64), \
                           (t-300)*600, 30000*tf.ones(tf.shape(t), dtype = tf.float64)))#Vaccination
        VCn = VCn*self.sf
        f_I = I_t - delta*delay + gamma*I
        f_J = J_t - (1-delta)*delay + gammaA*J
        f_D = D_t - (q*phiD)*H
        f_H = H_t - (p*gamma)*I + (q*phiD) * H + ((1-q)*phiR) * H
        f_R = R_t - gammaA*J - ((1-p)*gamma)*I - ((1-q)*phiR)*H - VCn*S/self.N
        f_I_sum = I_sum_t - delta*delay 
        f_H_sum = H_sum_t - (p*gamma)*I 

        return f_I, f_J, f_H, f_D, f_R, f_I_sum, f_H_sum  
        
    def callback(self, loss, lossU0, lossU, lossF, d1, d2, d3):
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF, d1, d2, d3]))
        print('L: %.3e, LU0: %.3e, LU: %.3e, LF: %.3e, d1:%.2f, d2:%.2f, d3:%.2f'  
              % (loss, lossU0, lossU, lossF, d1, d2, d3))        
   
    def train(self, nIter):  
                
        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f, 
                   self.I_new_u: self.I_new_train, self.D_new_u: self.D_new_train, self.H_new_u: self.H_new_train, 
                   self.I_sum_u: self.I_sum_train, self.H_sum_u: self.H_sum_train, self.D_sum_u: self.D_sum_train,  
                   self.S0_u: self.S0, self.E0_u: self.E0, self.I0_u: self.I0, 
                   self.J0_u: self.J0, self.D0_u: self.D0, self.H0_u: self.H0, 
                   self.R0_u: self.R0}
        
        start_time = time.time()
        for it in range(nIter+1):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lossU0_value = self.sess.run(self.lossU0, tf_dict)
                lossU_value = self.sess.run(self.lossU, tf_dict)
                lossF_value = self.sess.run(self.lossF, tf_dict)   
                d1_value=self.sess.run(self.d1, tf_dict)
                d2_value=self.sess.run(self.d2, tf_dict)
                d3_value=self.sess.run(self.d3, tf_dict)
                total_records.append(np.array([it, loss_value, lossU0_value, lossU_value, lossF_value, d1_value, d2_value, d3_value]))
                print('It: %d, L: %.3e, LU0: %.3e, LU: %.3e, LF: %.3e, d1:%.2f, d2:%.2f, d3:%.2f, Time: %.2f' % 
                      (it, loss_value, lossU0_value, lossU_value, lossF_value, d1_value, d2_value, d3_value, elapsed))
                start_time = time.time()
                
        if LBFGS:
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict, #Inputs of the minimize operator 
                                    fetches = [self.loss, self.lossU0, self.lossU, self.lossF, \
                                               self.d1, self.d2, self.d3,], 
                                    loss_callback = self.callback) #Show the results of minimize operator
            
    def predict(self, t_star):   
        
        tf_dict = {self.t_u: t_star}
        
        S = self. sess.run(self.S_pred, tf_dict) 
        I = self. sess.run(self.I_pred, tf_dict)
        J = self. sess.run(self.J_pred, tf_dict)
        H = self. sess.run(self.H_pred, tf_dict)
        D = self. sess.run(self.D_pred, tf_dict)
        R = self. sess.run(self.R_pred, tf_dict)
        I_sum = self. sess.run(self.I_sum_pred, tf_dict)
        D_sum = self. sess.run(self.D_sum_pred, tf_dict)
        H_sum = self. sess.run(self.H_sum_pred, tf_dict)  
        I_new = self. sess.run(self.I_new_pred, tf_dict)
        D_new = self. sess.run(self.D_new_pred, tf_dict)
        H_new = self. sess.run(self.H_new_pred, tf_dict) 
        BetaI = self. sess.run(self.BetaI_pred, tf_dict)  
        p = self. sess.run(self.p_pred, tf_dict) 
        q = self. sess.run(self.q_pred, tf_dict) 
        return S, I, J, D, H, R, I_new, D_new, H_new, I_sum, D_sum, H_sum, BetaI, p, q 

############################################################    
if __name__=="__main__": 
#%%  
    
    #Architecture of of the NN 
    layers=[1] + 5*[20] + [7] #The inout is t while the outputs are I,J,D,H,R,I_sum,H_sum,D_sum 
    layers_beta=[1] + 1*[5] + [1] 
    layers_p=[1] + 1*[5] + [1]
    layers_q=[1] + 1*[5] + [1]

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
    
    #lower and upper bounds 
    lb = t_star.min(0)
    ub = t_star.max(0) 
    
    #Initial conditions 
    I0_new = I_new_star[0:1,:] 
    D0_new = D_new_star[0:1,:]
    H0_new = H_new_star[0:1,:]
    I0_sum = I_sum_star[0:1,:] 
    D0_sum = D_sum_star[0:1,:]
    H0_sum = H_sum_star[0:1,:] 
    
    #Scaling 
    sf = 1e-4  
    N = N * sf  
    I_new_star = I_new_star * sf 
    H_new_star = H_new_star * sf 
    D_new_star = D_new_star * sf 
    I_sum_star = I_sum_star * sf  
    H_sum_star = H_sum_star * sf
    D_sum_star = D_sum_star * sf
    
    #Initial conditions 
    E0 = I_sum_star[0:1,:]
    I0 = I_sum_star[0:1,:]
    J0 = E0 - I0
    D0 = D_sum_star[0:1,:]
    H0 = np.array([[0.0]])
    R0 = np.array([[0.0]])
    S0 = N - E0 - I0 - J0 - D0 - H0 - R0
    S0 = S0.reshape([1,1]) 
    U0 = [S0, E0, I0, J0, D0, H0, R0]
    
    #Residual points 
    N_f = 3000
    t_f = lb + (ub-lb)*lhs(1, N_f)
    
#%%
    ######################################################################
    ######################## Training and Predicting ###############################
    ###################################################################### 
    t_train = t_star  
    I_new_train = I_new_star 
    D_new_train = D_new_star 
    H_new_train = H_new_star
    I_sum_train = I_sum_star 
    D_sum_train = D_sum_star 
    H_sum_train = H_sum_star 

    from datetime import datetime
    now = datetime.now()
    # dt_string = now.strftime("%m-%d-%H-%M")
    dt_string = now.strftime("%m-%d")
    
    
    
    #save results  
    current_directory = os.getcwd()
    for j in range(10):
        casenumber = 'set' + str(j+1)

        relative_path_results = '/Model7/Train-Results-'+dt_string+'-'+casenumber+'/'
        save_results_to = current_directory + relative_path_results
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        relative_path = '/Model7/Train-model-'+dt_string+'-'+casenumber+'/' 
        save_models_to = current_directory + relative_path
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)
            break
        
            #Training 
            total_records = []
            total_records_LBFGS = []
            model = PhysicsInformedNN(t_train, I_new_train, D_new_train, H_new_train, 
                         I_sum_train, D_sum_train, H_sum_train, U0, t_f, lb, ub, N, 
                         layers, layers_beta, layers_p, layers_q, sf)
        
            LBFGS = True
            # LBFGS = False
            model.train(10000) #Training with n iterations 
        
            ####save model 
            model.saver.save(model.sess, save_models_to+"model.ckpt")    
        
            #Predicting   
            S, I, J, D, H, R, I_new, D_new, H_new, I_sum, D_sum, H_sum, BetaI, p, q  = model.predict(t_star) 
        
            #Calculate RC  
            Rc = BetaI * ((1.0-0.6)*0.75/(1.0/6.0) + 0.6/(1.0/6.0))   
            
            d1_value=model.sess.run(model.d1)
            d2_value=model.sess.run(model.d2)
            d3_value=model.sess.run(model.d3) 
            
            print('d1: %.3f'% (d1_value))
            print('d2: %.3f'% (d2_value))
            print('d3: %.3f'% (d3_value))
            
            import datetime
            end_time = time.time()
            print(datetime.timedelta(seconds=int(end_time-start_time)))
             
        #%%    
            #save data
            np.savetxt(save_results_to +'S.txt', S.reshape((-1,1))) 
            np.savetxt(save_results_to +'I.txt', I.reshape((-1,1)))
            np.savetxt(save_results_to +'J.txt', J.reshape((-1,1)))
            np.savetxt(save_results_to +'D.txt', D.reshape((-1,1)))
            np.savetxt(save_results_to +'H.txt', H.reshape((-1,1)))
            np.savetxt(save_results_to +'R.txt', R.reshape((-1,1)))
            np.savetxt(save_results_to +'I_new.txt', I_new.reshape((-1,1)))
            np.savetxt(save_results_to +'D_new.txt', D_new.reshape((-1,1)))
            np.savetxt(save_results_to +'H_new.txt', H_new.reshape((-1,1)))
            np.savetxt(save_results_to +'I_sum.txt', I_sum.reshape((-1,1))) 
            np.savetxt(save_results_to +'H_sum.txt', H_sum.reshape((-1,1))) 
            np.savetxt(save_results_to +'D_sum.txt', D_sum.reshape((-1,1)))  
            #save BetaI, Rc, and q
            np.savetxt(save_results_to +'t_star.txt', t_star.reshape((-1,1))) 
            np.savetxt(save_results_to +'BetaI.txt', BetaI.reshape((-1,1)))
            np.savetxt(save_results_to +'Rc.txt', Rc.reshape((-1,1)))   
            np.savetxt(save_results_to +'p.txt', p.reshape((-1,1)))
            np.savetxt(save_results_to +'q.txt', q.reshape((-1,1)))
        
        #%% 
            #records for Adam
            N_Iter = len(total_records)
            iteration = np.asarray(total_records)[:,0]
            loss_his = np.asarray(total_records)[:,1]
            loss_his_u0  = np.asarray(total_records)[:,2]
            loss_his_u  = np.asarray(total_records)[:,3]
            loss_his_f  = np.asarray(total_records)[:,4]  
            loss_his_d1  = np.asarray(total_records)[:,5]  
            loss_his_d2  = np.asarray(total_records)[:,6]  
            loss_his_d3  = np.asarray(total_records)[:,7]   
            
            #records for LBFGS
            if LBFGS:
                N_Iter_LBFGS = len(total_records_LBFGS)
                iteration_LBFGS = np.arange(N_Iter_LBFGS)+N_Iter*100
                loss_his_LBFGS = np.asarray(total_records_LBFGS)[:,0]
                loss_his_u0_LBFGS = np.asarray(total_records_LBFGS)[:,1]
                loss_his_u_LBFGS  = np.asarray(total_records_LBFGS)[:,2]
                loss_his_f_LBFGS  = np.asarray(total_records_LBFGS)[:,3]
                loss_his_d1_LBFGS  = np.asarray(total_records_LBFGS)[:,4]
                loss_his_d2_LBFGS  = np.asarray(total_records_LBFGS)[:,5]
                loss_his_d3_LBFGS  = np.asarray(total_records_LBFGS)[:,6]
        
        #%%    
            #save records
            np.savetxt(save_results_to +'iteration.txt', iteration.reshape((-1,1))) 
            np.savetxt(save_results_to +'loss_his.txt', loss_his.reshape((-1,1)))
            np.savetxt(save_results_to +'loss_his_u0.txt', loss_his_u0.reshape((-1,1))) 
            np.savetxt(save_results_to +'loss_his_u.txt', loss_his_u.reshape((-1,1))) 
            np.savetxt(save_results_to +'loss_his_f.txt', loss_his_f.reshape((-1,1))) 
            np.savetxt(save_results_to +'loss_his_d1.txt', loss_his_d1.reshape((-1,1)))  
            np.savetxt(save_results_to +'loss_his_d2.txt', loss_his_d2.reshape((-1,1)))  
            np.savetxt(save_results_to +'loss_his_d3.txt', loss_his_d3.reshape((-1,1)))    
            
            if LBFGS:
                np.savetxt(save_results_to +'iteration_LBFGS.txt', iteration_LBFGS.reshape((-1,1))) 
                np.savetxt(save_results_to +'loss_his_LBFGS.txt', loss_his_LBFGS.reshape((-1,1)))
                np.savetxt(save_results_to +'loss_his_u0_LBFGS.txt', loss_his_u0_LBFGS.reshape((-1,1))) 
                np.savetxt(save_results_to +'loss_his_u_LBFGS.txt', loss_his_u_LBFGS.reshape((-1,1))) 
                np.savetxt(save_results_to +'loss_his_f_LBFGS.txt', loss_his_f_LBFGS.reshape((-1,1)))  
                np.savetxt(save_results_to +'loss_his_d1_LBFGS.txt', loss_his_d1_LBFGS.reshape((-1,1))) 
                np.savetxt(save_results_to +'loss_his_d2_LBFGS.txt', loss_his_d2_LBFGS.reshape((-1,1))) 
                np.savetxt(save_results_to +'loss_his_d3_LBFGS.txt', loss_his_d3_LBFGS.reshape((-1,1)))  
        
        #%%
            ######################################################################
            ############################# Plotting ###############################
            ###################################################################### 
        #%%    
            #History of loss 
            font = 24
            fig, ax = plt.subplots()
            plt.legend(loc="best",  fontsize = 24, ncol=4) 
            plt.locator_params(axis='x', nbins=6)
            plt.locator_params(axis='y', nbins=6)
            plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
            plt.xlabel('$iteration$', fontsize = font)
            plt.ylabel('$loss values$', fontsize = font)
            plt.yscale('log')
            plt.grid(True) 
            plt.plot(iteration,loss_his, label='$loss$')
            plt.plot(iteration,loss_his_u0, label='$loss_{u0}$')
            plt.plot(iteration,loss_his_u, label='$loss_u$')
            plt.plot(iteration,loss_his_f, label='$loss_f$')  
            if LBFGS:
                plt.plot(iteration_LBFGS,loss_his_LBFGS)
                plt.plot(iteration_LBFGS,loss_his_u0_LBFGS)
                plt.plot(iteration_LBFGS,loss_his_u_LBFGS)
                plt.plot(iteration_LBFGS,loss_his_f_LBFGS) 
            plt.legend(loc="upper right",  fontsize = 24, ncol=4)
            plt.legend()
            ax.tick_params(axis='both', labelsize = 24)
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'History_loss.png', dpi=300)  
            
        #%%    
            #History of parameters 
            lw = 2
            font = 24
            fig, ax = plt.subplots()
            plt.locator_params(axis='x', nbins=6)
            plt.locator_params(axis='y', nbins=6)
            plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
            plt.xlabel('$iteration$', fontsize = font)
            plt.ylabel('$Parameters$', fontsize = font) 
            plt.grid(True) 
            # plt.plot(iteration,loss_his_d1, 'k-', lw = lw, label='$d_{1}$')
            plt.plot(iteration,loss_his_d2, 'r-', lw = lw, label='$d_{2}$')
            # plt.plot(iteration,loss_his_d3, 'b--', lw = lw, label='$d_{3}$')   
            
            if LBFGS:
                # plt.plot(iteration_LBFGS,loss_his_d1_LBFGS, 'k-', lw = lw)
                plt.plot(iteration_LBFGS,loss_his_d2_LBFGS, 'r-', lw = lw)
                # plt.plot(iteration_LBFGS,loss_his_d3_LBFGS, 'b-', lw = lw) 
            
            plt.legend(loc="upper right",  fontsize = 24, ncol=4) 
            ax.tick_params(axis='both', labelsize = 24)
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'History_Parameters.png', dpi=300)  
            
        #%% 
            #Suspectious 
            font = 24
            fig, ax = plt.subplots() 
            ax.plot(t_star, S/sf, 'r-', lw=2, label='PINN') 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(6,6))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{S}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Current_Suspectious.png', dpi=300) 
            
        #%%    
            #Infections 
            font = 24
            fig, ax = plt.subplots() 
            ax.plot(t_star, I/sf, 'r-', lw=2, label='PINN') 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{I}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Current_Infections.png', dpi=300)  
        
        #%%    
            #Asymptomatic 
            font = 24
            fig, ax = plt.subplots() 
            ax.plot(t_star, J/sf, 'r-', lw=2, label='PINN') 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{J}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Current_Asymptomatic.png', dpi=300)  
        
        #%%
            #Hospitalizations
            font = 24
            fig, ax = plt.subplots() 
            ax.plot(t_star, H/sf, 'r-',lw=2,label='PINN') 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{H}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Current_Hospitalized.png', dpi=300)  
        
        #%%
            #Death 
            font = 24
            fig, ax = plt.subplots() 
            ax.plot(t_star, D/sf, 'r-',lw=2, label='PINN') 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{D}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Current_Death.png', dpi=300)     
            
        #%%
            #Recovered 
            font = 24
            fig, ax = plt.subplots() 
            ax.plot(t_star, R/sf, 'r-',lw=2, label='PINN') 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{R}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Current_Recovered.png', dpi=300)     
            
        #%%
            #New cases 
            #Confirmed Cases
            font = 24
            fig, ax = plt.subplots()
            ax.plot(t_star, I_new_star/sf, 'k--', marker = 'o', lw=2, markersize=5, label='Data-7davg') 
            ax.plot(t_star[:-1], I_new/sf, 'r-', lw=2, label='PINN-training')  
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{I}^{new}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'new_cases.png', dpi=300)  
        
        #%%
            #Hospitalized Cases
            font = 24
            fig, ax = plt.subplots()
            ax.plot(t_star, H_new_star/sf, 'k--', marker = 'o', lw=2, markersize=5, label='Data-7davg') 
            ax.plot(t_star[:-1], H_new/sf, 'r-', lw=2, label='PINN-training')  
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{H}^{new}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'new_hospitalized.png', dpi=300)  
        
        #%%
            #Death Cases
            font = 24
            fig, ax = plt.subplots()
            ax.plot(t_star, D_new_star/sf, 'k--', marker = 'o', lw=2, markersize=5, label='Data-7davg') 
            ax.plot(t_star[:-1], D_new/sf, 'r-', lw=2, label='PINN-training')  
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{D}^{new}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'new_death.png', dpi=300)  
            
        #%%
            #Accumulative confirmed Cases
            font = 24
            fig, ax = plt.subplots()
            ax.plot(t_star, I_sum_star/sf, 'k--', marker = 'o', lw=2, markersize=5, label='Data-7davg') 
            ax.plot(t_star, I_sum/sf, 'r-', lw=2,label='PINN-training')  
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{I}^{c}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Cumulative_cases.png', dpi=300)  
        
        #%%
            #Accumulative hospitalized Cases
            font = 24
            fig, ax = plt.subplots()
            ax.plot(t_star, H_sum_star/sf, 'k--', marker = 'o', lw=2, markersize=5, label='Data-7davg') 
            ax.plot(t_star, H_sum/sf, 'r-', lw=2,label='PINN-training')  
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{H}^{c}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Cumulative_hospitalized.png', dpi=300)  
        
        #%%
            #Accumulative death cases
            font = 24
            fig, ax = plt.subplots()
            ax.plot(t_star, D_sum_star/sf, 'k--', marker = 'o', lw=2, markersize=5, label='Data-7davg') 
            ax.plot(t_star, D_sum/sf, 'r-', lw=2,label='PINN-training')  
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$\mathbf{D}^{c}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Cumulative_death.png', dpi=300) 
        
            
        #%%
            #BetaI curve 
            font = 24
            fig, ax = plt.subplots()  
            ax.plot(t_star, BetaI, 'r-',lw=2) 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$Beta_{I}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'BetaI.png', dpi=300)  
            
        #%%   
            #RC curve 
            font = 24
            fig, ax = plt.subplots()  
            ax.plot(t_star, Rc, 'r-',lw=2) 
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$R_{c}$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'Rc.png', dpi=300)
        
        #%%   
            #p curve 
            font = 24
            fig, ax = plt.subplots()  
            ax.plot(t_star, p, 'r-', lw=2, label='PINN')    
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$p$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'p.png', dpi=300)
            
        #%%   
            #q curve 
            font = 24
            fig, ax = plt.subplots()  
            ax.plot(t_star, q, 'r-', lw=2, label='PINN')    
            # ax.set_xlim(0-0.5,180)
            # ax.set_ylim(0-0.5,6000+0.5)
            # ax.legend(fontsize=22)
            ax.tick_params(axis='both', labelsize = 24)
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.grid(True)
            ax.set_xlabel('Days', fontsize = font)
            ax.set_ylabel('$q$', fontsize = font) 
            fig.set_size_inches(w=13,h=6.5)
            plt.savefig(save_results_to +'q.png', dpi=300)
            
        
             
