# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:18:35 2020

@author: Administrator 

Solve the inverse problem of the coupled integer-order ODE system  
#S = N-E-P-I-A-D-H-Q-R
dSdt = - (BetaI(t)*(I+eps*A+eps*P)/N) * S
dEdt = (BetaI(t)*(I+eps*A+eps*P)/N) * S - d_E * E
dPdt = d_E * E - d_P * P
dIdt = (delta*d_P) * P - d_I * I
dAdt = ((1-delta)*d_P) * P - d_A * A
dHdt = (p*d_I) * I - d_H * H
dQdt = ((1-p)*d_I) * I - d_Q * Q
dDdt = (q*d_H) * H
dRdt = d_A * A + ((1-q)*d_H) * H + d_Q * Q
dsumIdt = (delta*d_P) * P
dsumHdt = (p*d_I) * I

Here the parameters BetaI(t) p(t), and q(t) are inferred by PINNs

The fixed parameters are  


"""


import sys
sys.path.insert(0, '../../Utilities/')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas
from scipy.integrate import odeint
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
class PhysicsInformedNN:
    #Initialize the class
    def __init__(self, t_train, I_new_train, D_new_train, H_train, 
                 I_sum_train, D_sum_train, U0, t_f, lb, ub, N, 
                 layers, layers_beta, layers_p, layers_q, sf): 
        
        self.N = N
        self.sf = sf
        
        #Data for training  
        self.t_train = t_train  
        self.I_new_train = I_new_train 
        self.D_new_train = D_new_train  
        self.H_train = H_train
        self.I_sum_train = I_sum_train 
        self.D_sum_train = D_sum_train  
        self.I0_new = U0[0] 
        self.D0_new = U0[1] 
        self.H0 = U0[2]
        self.I0_sum = U0[3] 
        self.D0_sum = U0[4]  
        self.J0 = U0[5]  
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
        self.eps = tf.Variable(0.75,dtype=tf.float64,trainable=False) 
        self.delta = tf.Variable(0.6,dtype=tf.float64,trainable=False) 
        self.d_E = tf.Variable(1.0/2.9,dtype=tf.float64,trainable=False) 
        self.d_P = tf.Variable(1.0/2.3,dtype=tf.float64,trainable=False) 
        self.d_I = tf.Variable(1.0/2.9,dtype=tf.float64,trainable=False) 
        # self.d_A = tf.Variable(1.0/7,dtype=tf.float64,trainable=False) 
        # self.d_H = tf.Variable(1.0/6.9,dtype=tf.float64,trainable=False)
        self.d_A = tf.Variable(1.0/7.5,dtype=tf.float64,trainable=False) 
        self.d_H = tf.Variable(1.0/15.0,dtype=tf.float64,trainable=False)  
        self.d_Q = tf.Variable(1.0/10,dtype=tf.float64,trainable=False)  

       # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.saver = tf.train.Saver()
        

        # placeholders for inputs  
        self.t_u = tf.placeholder(tf.float64, shape=[None, self.t_train.shape[1]])  
        self.I_new_u = tf.placeholder(tf.float64, shape=[None, self.I_new_train.shape[1]]) 
        self.D_new_u = tf.placeholder(tf.float64, shape=[None, self.D_new_train.shape[1]])  
        self.H_u = tf.placeholder(tf.float64, shape=[None, self.H_train.shape[1]]) 
        self.I_sum_u = tf.placeholder(tf.float64, shape=[None, self.I_sum_train.shape[1]]) 
        self.D_sum_u = tf.placeholder(tf.float64, shape=[None, self.D_sum_train.shape[1]])  
        self.I0_new_u = tf.placeholder(tf.float64, shape=[None, self.I0_new.shape[1]])
        self.D0_new_u = tf.placeholder(tf.float64, shape=[None, self.D0_new.shape[1]]) 
        self.H0_u = tf.placeholder(tf.float64, shape=[None, self.H0.shape[1]])  
        self.I0_sum_u = tf.placeholder(tf.float64, shape=[None, self.I0_sum.shape[1]])
        self.D0_sum_u = tf.placeholder(tf.float64, shape=[None, self.D0_sum.shape[1]]) 
        self.J0_u = tf.placeholder(tf.float64, shape=[None, self.J0.shape[1]]) 
        self.R0_u = tf.placeholder(tf.float64, shape=[None, self.R0.shape[1]]) 
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])   

        # physics informed neural networks 
        self.S_pred, self.E_pred, self.PreS_pred, self.I_pred, self.J_pred, self.Qua_pred, self.D_pred, self.H_pred, self.R_pred, self.I_sum_pred, self.H_sum_pred, self.D_sum_pred = self.net_u(self.t_u)
        
        self.BetaI_pred = self.net_BetaI(self.t_u)          
        self.p_pred = self.net_p(self.t_u)  
        self.q_pred = self.net_q(self.t_u)  
        
        self.I_new_pred = self.I_sum_pred[1:,:]-self.I_sum_pred[0:-1,:]
        self.D_new_pred = self.D_sum_pred[1:,:]-self.D_sum_pred[0:-1,:]
        
        self.H0_pred = self.H_pred[0]
        self.I0_sum_pred = self.I_sum_pred[0] 
        self.D0_sum_pred = self.D_sum_pred[0]
        self.H0_sum_pred = self.H_sum_pred[0]
        self.J0_pred = self.J_pred[0]
        self.R0_pred = self.R_pred[0]
        
        self.E_f, self.PreS_f, self.I_f, self.J_f, self.H_f, self.Qua_f, self.D_f, self.R_f, self.I_sum_f, self.H_sum_f = self.net_f(self.t_tf)
        
        # # loss
        # self.lossU0 = tf.reduce_mean(tf.square(self.H0_u - self.H0_pred)) + \
        #     tf.reduce_mean(tf.square(self.I0_sum_u - self.I0_sum_pred)) + \
        #     100*tf.reduce_mean(tf.square(self.D0_sum_u - self.D0_sum_pred)) + \
        #     tf.reduce_mean(tf.square(self.H0_u - self.H0_sum_pred)) + \
        #     tf.reduce_mean(tf.square(self.J0_u - self.J0_pred)) + \
        #     tf.reduce_mean(tf.square(self.R0_u - self.R0_pred))   
        
        # self.lossU = tf.reduce_mean(tf.square(self.I_sum_u - self.I_sum_pred)) + \
        #     100*tf.reduce_mean(tf.square(self.D_sum_u - self.D_sum_pred)) + \
        #     tf.reduce_mean(tf.square(self.H_u - self.H_pred)) + \
        #     tf.reduce_mean(tf.square(self.I_new_u[1:,:] - self.I_new_pred)) + \
        #     100*tf.reduce_mean(tf.square(self.D_new_u[1:,:] - self.D_new_pred))
        
        # self.lossF = tf.reduce_mean(tf.square(self.E_f)) + tf.reduce_mean(tf.square(self.I_f)) + \
        #     tf.reduce_mean(tf.square(self.J_f)) + tf.reduce_mean(tf.square(self.D_f)) + \
        #     tf.reduce_mean(tf.square(self.H_f)) + tf.reduce_mean(tf.square(self.R_f)) + \
        #     tf.reduce_mean(tf.square(self.I_sum_f)) + tf.reduce_mean(tf.square(self.H_sum_f)) + \
        #     tf.reduce_mean(tf.square(self.PreS_f)) + tf.reduce_mean(tf.square(self.Qua_f))

        # self.lossQ = 100*tf.reduce_mean(tf.square((self.D_new_u/self.H_u)/self.d_H- self.q_pred))

        # self.loss = self.lossU0 + self.lossU + self.lossF  + self.lossQ 
        
        # loss
        self.lossU0 = tf.reduce_mean(tf.square(self.H0_u - self.H0_pred)) + \
            tf.reduce_mean(tf.square(self.I0_sum_u - self.I0_sum_pred)) + \
            tf.reduce_mean(tf.square(self.D0_sum_u - self.D0_sum_pred)) + \
            tf.reduce_mean(tf.square(self.H0_u - self.H0_sum_pred)) + \
            tf.reduce_mean(tf.square(self.J0_u - self.J0_pred)) + \
            tf.reduce_mean(tf.square(self.R0_u - self.R0_pred)) 
            
        self.lossU = tf.reduce_mean(tf.square(self.I_sum_u - self.I_sum_pred)) + \
            60*tf.reduce_mean(tf.square(self.D_sum_u - self.D_sum_pred)) + \
            20*tf.reduce_mean(tf.square(self.H_u - self.H_pred)) + \
            50*tf.reduce_mean(tf.square(self.I_new_u[1:,:] - self.I_new_pred)) + \
            6000*tf.reduce_mean(tf.square(self.D_new_u[1:,:] - self.D_new_pred))
        
        self.lossF = tf.reduce_mean(tf.square(self.E_f)) + tf.reduce_mean(tf.square(self.I_f)) + \
            tf.reduce_mean(tf.square(self.J_f)) + tf.reduce_mean(tf.square(self.D_f)) + \
            tf.reduce_mean(tf.square(self.H_f)) + tf.reduce_mean(tf.square(self.R_f)) + \
            tf.reduce_mean(tf.square(self.I_sum_f)) + tf.reduce_mean(tf.square(self.H_sum_f)) + \
            tf.reduce_mean(tf.square(self.PreS_f)) + tf.reduce_mean(tf.square(self.Qua_f))

        self.lossQ = 100*tf.reduce_mean(tf.square((self.D_new_u/self.H_u)/self.d_H- self.q_pred)) 

        self.loss = self.lossU0 + self.lossU + self.lossF  + self.lossQ


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
            W = self.xavier_init(size=[layers[l], layers[l+1]]) #weights for the current layer 
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64) #biases for the current layer
            weights.append(W) #save the elements in W to weights (a row vector) 
            biases.append(b)   #save the elements in b to biases (a 1Xsum(layers) row vector)      
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
    
    def net_BetaI(self,t):
        BetaI = self.neural_net(t, self.weights_beta, self.biases_beta) 
        return tf.sigmoid(BetaI) 
    
    def net_p(self,t):
        p = self.neural_net(t, self.weights_p, self.biases_p)
        return tf.sigmoid(p) 
    
    def net_q(self,t):
        q = self.neural_net(t, self.weights_q, self.biases_q)
        return tf.sigmoid(q) 
    
    def net_u(self, t):
        SEPIAHQDR = self.neural_net(t, self.weights, self.biases)  
        E = SEPIAHQDR[:,0:1] 
        PreS = SEPIAHQDR[:,1:2]
        I = SEPIAHQDR[:,2:3]
        A = SEPIAHQDR[:,3:4]
        H = SEPIAHQDR[:,4:5]
        Qua = SEPIAHQDR[:,5:6]     
        D = SEPIAHQDR[:,6:7]
        R = SEPIAHQDR[:,7:8] 
        I_sum = SEPIAHQDR[:,8:9]
        H_sum = SEPIAHQDR[:,9:10] 
        D_sum = D
        S = self.N-E-PreS-I-A-H-Qua-D-R  
        return S, E, PreS, I, A, Qua, D, H, R, I_sum, H_sum, D_sum 
    
    def net_f(self, t):  
        #load time-dependent parameters
        betaI = self.net_BetaI(t)  
        p = self.net_p(t)
        q = self.net_q(t)   
        
        #Obtain S,E,I,J,D,H,R from Neural network
        S, E, PreS, I, A, Qua, D, H, R, I_sum, H_sum, D_sum  = self.net_u(t)   
        
        #Time derivatives  
        E_t = tf.gradients(E, t, unconnected_gradients='zero')[0]
        PreS_t = tf.gradients(PreS, t, unconnected_gradients='zero')[0]
        I_t = tf.gradients(I, t, unconnected_gradients='zero')[0]
        A_t = tf.gradients(A, t, unconnected_gradients='zero')[0]
        H_t = tf.gradients(H, t, unconnected_gradients='zero')[0]
        Qua_t = tf.gradients(Qua, t, unconnected_gradients='zero')[0]
        D_t = tf.gradients(D, t, unconnected_gradients='zero')[0]
        R_t = tf.gradients(R, t, unconnected_gradients='zero')[0]
        I_sum_t = tf.gradients(I_sum, t, unconnected_gradients='zero')[0]
        H_sum_t = tf.gradients(H_sum, t, unconnected_gradients='zero')[0] 
        
        #Residuals  
        VCn = tf.where(t<260*tf.ones(tf.shape(t), dtype = tf.float64), \
                       tf.zeros(tf.shape(t), dtype = tf.float64), (t-260)*450)#Vaccination
        VCn = VCn*self.sf
        f_E = E_t - (betaI*(I+self.eps*A+self.eps*PreS)/N) * S + self.d_E * E
        f_PreS = PreS_t - self.d_E * E + self.d_P * PreS
        f_I = I_t - (self.delta*self.d_P) * PreS + self.d_I * I
        f_A = A_t - ((1-self.delta)*self.d_P) * PreS + self.d_A * A
        f_H = H_t - (p*self.d_I) * I + self.d_H * H
        f_Qua = Qua_t - ((1-p)*self.d_I) * I + self.d_Q * Qua
        f_D = D_t - (q*self.d_H) * H
        f_R = R_t - self.d_A * A - ((1-q)*self.d_H) * H - self.d_Q * Qua - VCn*S/self.N
        f_I_sum = I_sum_t - (self.delta*self.d_P) * PreS
        f_H_sum = H_sum_t - (p*self.d_I) * I
        
        return f_E, f_PreS, f_I, f_A, f_H, f_Qua, f_D, f_R, f_I_sum, f_H_sum


    def callback(self, loss, lossU0, lossU, lossF, lossQ):
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF, lossQ]))
        print('Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e, LossQ: %.3e'  
              % (loss, lossU0, lossU, lossF, lossQ))        
   
    def train(self, nIter):  
                
        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f, 
                   self.I_new_u: self.I_new_train, self.D_new_u: self.D_new_train, self.H_u: self.H_train, 
                   self.I_sum_u: self.I_sum_train, self.D_sum_u: self.D_sum_train,  
                   self.I0_new_u: self.I0_new, self.D0_new_u: self.D0_new, self.H0_u: self.H0, 
                   self.I0_sum_u: self.I0_sum, self.D0_sum_u: self.D0_sum, self.J0_u: self.J0, self.R0_u: self.R0}
        
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
                lossQ_value = self.sess.run(self.lossQ, tf_dict)   
                total_records.append(np.array([it, loss_value, lossU0_value, lossU_value, lossF_value, lossQ_value]))
                print('It: %d, Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e, LossQ: %.3e, Time: %.2f' % 
                      (it, loss_value, lossU0_value, lossU_value, lossF_value, lossQ_value, elapsed))
                start_time = time.time()
                
        if LBFGS:     
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict, #Inputs of the minimize operator 
                                    fetches = [self.loss, self.lossU0, self.lossU, self.lossF, self.lossQ], 
                                    loss_callback = self.callback) #Show the results of minimize operator

    def predict(self, t_star):        

        tf_dict = {self.t_u: t_star, self.I0_new_u: self.I0_new, \
                   self.D0_new_u: self.D0_new, self.J0_u: self.J0, self.R0_u: self.R0} 
        S = self. sess.run(self.S_pred, tf_dict)
        E = self. sess.run(self.E_pred, tf_dict)
        PreS = self. sess.run(self.PreS_pred, tf_dict)
        I = self. sess.run(self.I_pred, tf_dict)
        J = self. sess.run(self.J_pred, tf_dict)
        Qua = self. sess.run(self.Qua_pred, tf_dict)
        D = self. sess.run(self.D_pred, tf_dict)
        H = self. sess.run(self.H_pred, tf_dict)
        R = self. sess.run(self.R_pred, tf_dict)
        I_sum = self. sess.run(self.I_sum_pred, tf_dict)
        D_sum = self. sess.run(self.D_sum_pred, tf_dict)
        H_sum = self. sess.run(self.H_sum_pred, tf_dict)  
        I_new = self. sess.run(self.I_new_pred, tf_dict)
        D_new = self. sess.run(self.D_new_pred, tf_dict) 
        H_new = np.diff(H_sum,axis=0)
        BetaI = self. sess.run(self.BetaI_pred, tf_dict)  
        p = self. sess.run(self.p_pred, tf_dict) 
        q = self. sess.run(self.q_pred, tf_dict) 
        return S, E, PreS, I, J, Qua, D, H, R, I_new, D_new, H_new, I_sum, D_sum, H_sum, BetaI, p, q 
        

#%%
############################################################    
if __name__=="__main__":  
#%%    
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
    
    #Scaling 
    sf = 1e-4 
    N = N * sf  
    I_new_star = I_new_star * sf 
    D_new_star = D_new_star * sf
    H_star = H_star * sf  
    I_sum_star = I_sum_star * sf  
    D_sum_star = D_sum_star * sf
    
    #Initial conditions 
    I0_new = I_new_star[0:1,:] 
    D0_new = D_new_star[0:1,:]
    H0 = H_star[0:1,:]
    I0_sum = I_sum_star[0:1,:] 
    D0_sum = D_sum_star[0:1,:]   
    J0 = np.array([[0.0]])
    R0 = np.array([[0.0]])
    U0 = [I0_new, D0_new, H0, I0_sum, D0_sum, J0, R0]
    
#%%
    ######################################################################
    ######################## Training and Predicting ###############################
    ######################################################################  
    #Architecture of of the NN 
    layers=[1] + 10*[32] + [10] 
    layers_beta=[1] + 1*[5] + [1] 
    layers_p=[1] + 1*[5] + [1]
    layers_q=[1] + 10*[32] + [1]    
    
    #Data for training  
    I_new_train = I_new_star 
    D_new_train = D_new_star 
    H_train = H_star  
    I_sum_train = I_sum_star 
    D_sum_train = D_sum_star 
    t_train = t_star  
    
    #lower and upper bounds 
    lb = t_train.min(0)
    ub = t_train.max(0) 
    
    #Residual points 
    N_f = 1000
    # t_f = lb + (ub-lb)*lhs(1, N_f) #uniformed grid for evaluating fractional derivative 
    t_f = np.linspace(lb, ub, num = N_f)
    
    from datetime import datetime
    now = datetime.now()
    # dt_string = now.strftime("%m-%d-%H-%M")
    dt_string = now.strftime("%m-%d")
    # dt_string = '05-04'

    #save results  
    current_directory = os.getcwd()
    for j in range(3):
        casenumber = 'set' + str(j+1)

        # relative_path_results = '/Model3/Train-Results-'+casenumber+'/'
        relative_path_results = '/Model3/Train-Results-'+dt_string+'-'+casenumber+'/'
        save_results_to = current_directory + relative_path_results
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        # relative_path = '/Model3/Train-model-'+casenumber+'/'
        relative_path = '/Model3/Train-model-'+dt_string+'-'+casenumber+'/' 
        save_models_to = current_directory + relative_path
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)
            # break
        
        #Training 
        total_records = []
        total_records_LBFGS = []
        model = PhysicsInformedNN(t_train, I_new_train, D_new_train, H_train, 
                     I_sum_train, D_sum_train, U0, t_f, lb, ub, N, 
                     layers, layers_beta, layers_p, layers_q, sf)
    
        LBFGS = True
        # LBFGS = False
        model.train(10000) #Training with n iterations 
        
    #%%        
        #save model  
        model.saver.save(model.sess, save_models_to+"model.ckpt")
    
        #Predicting   
        S, E, PreS, I, J, Qua, D, H, R, I_new, D_new, H_new, I_sum, D_sum, H_sum, BetaI, p, q= model.predict(t_train) 
        import datetime
        end_time = time.time()
        print(datetime.timedelta(seconds=int(end_time-start_time)))
             
        #Calculate RC  
        Rc = BetaI * ((1.0-0.6)*0.75/(1.0/6.0) + 0.6/(1.0/6.0) + 0.55/(1.0/2.3))   
            
        #save data
        np.savetxt(save_results_to +'S.txt', S.reshape((-1,1)))
        np.savetxt(save_results_to +'E.txt', E.reshape((-1,1)))
        np.savetxt(save_results_to +'PreS.txt', PreS.reshape((-1,1)))
        np.savetxt(save_results_to +'I.txt', I.reshape((-1,1)))
        np.savetxt(save_results_to +'J.txt', J.reshape((-1,1)))
        np.savetxt(save_results_to +'Qua.txt', Qua.reshape((-1,1)))
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
        np.savetxt(save_results_to +'t_train.txt', t_train.reshape((-1,1))) 
        np.savetxt(save_results_to +'BetaI.txt', BetaI.reshape((-1,1)))
        np.savetxt(save_results_to +'Rc.txt', Rc.reshape((-1,1)))   
        np.savetxt(save_results_to +'p.txt', p.reshape((-1,1)))
        np.savetxt(save_results_to +'q.txt', q.reshape((-1,1)))
     
        #records for Adam
        N_Iter = len(total_records)
        iteration = np.asarray(total_records)[:,0]
        loss_his = np.asarray(total_records)[:,1]
        loss_his_u0  = np.asarray(total_records)[:,2]
        loss_his_u  = np.asarray(total_records)[:,3]
        loss_his_f  = np.asarray(total_records)[:,4] 
        loss_his_q  = np.asarray(total_records)[:,5]     
        
        #records for LBFGS
        if LBFGS:
            N_Iter_LBFGS = len(total_records_LBFGS)
            iteration_LBFGS = np.arange(N_Iter_LBFGS)+N_Iter*100
            loss_his_LBFGS = np.asarray(total_records_LBFGS)[:,0]
            loss_his_u0_LBFGS = np.asarray(total_records_LBFGS)[:,1]
            loss_his_u_LBFGS  = np.asarray(total_records_LBFGS)[:,2]
            loss_his_f_LBFGS  = np.asarray(total_records_LBFGS)[:,3]
            loss_his_q_LBFGS  = np.asarray(total_records_LBFGS)[:,4]
        
        #save records
        np.savetxt(save_results_to +'iteration.txt', iteration.reshape((-1,1))) 
        np.savetxt(save_results_to +'loss_his.txt', loss_his.reshape((-1,1)))
        np.savetxt(save_results_to +'loss_his_u0.txt', loss_his_u0.reshape((-1,1))) 
        np.savetxt(save_results_to +'loss_his_u.txt', loss_his_u.reshape((-1,1))) 
        np.savetxt(save_results_to +'loss_his_f.txt', loss_his_f.reshape((-1,1)))  
        np.savetxt(save_results_to +'loss_his_q.txt', loss_his_q.reshape((-1,1)))   
        
        if LBFGS:
            np.savetxt(save_results_to +'iteration_LBFGS.txt', iteration_LBFGS.reshape((-1,1))) 
            np.savetxt(save_results_to +'loss_his_LBFGS.txt', loss_his_LBFGS.reshape((-1,1)))
            np.savetxt(save_results_to +'loss_his_u0_LBFGS.txt', loss_his_u0_LBFGS.reshape((-1,1))) 
            np.savetxt(save_results_to +'loss_his_u_LBFGS.txt', loss_his_u_LBFGS.reshape((-1,1))) 
            np.savetxt(save_results_to +'loss_his_f_LBFGS.txt', loss_his_f_LBFGS.reshape((-1,1)))  
            np.savetxt(save_results_to +'loss_his_q_LBFGS.txt', loss_his_q_LBFGS.reshape((-1,1)))   
        
        ######################################################################
        ############################# Plotting ###############################
        ###################################################################### 
        SAVE_FIG = True
    
        #History of loss 
        font = 24
        fig, ax = plt.subplots()
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
        plt.plot(iteration,loss_his_q, label='$loss_q$')  
        if LBFGS: 
            plt.plot(iteration_LBFGS,loss_his_LBFGS, label='$loss-LBFGS$')
            plt.plot(iteration_LBFGS,loss_his_u0_LBFGS, label='$loss_{u0}-LBFGS$')
            plt.plot(iteration_LBFGS,loss_his_u_LBFGS, label='$loss_u-LBFGS$')
            plt.plot(iteration_LBFGS,loss_his_f_LBFGS, label='$loss_f-LBFGS$') 
            plt.plot(iteration_LBFGS,loss_his_q_LBFGS, label='$loss_q-LBFGS$') 
        plt.legend(loc="upper right",  fontsize = 24, ncol=4)
        plt.legend()
        ax.tick_params(axis='both', labelsize = 24)
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'History_loss.png', dpi=300)    
         
        #Current cases     
        #Infections 
        font = 24
        fig, ax = plt.subplots() 
        ax.plot(t_star, I/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('Current infections ($I$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Current_infections.png', dpi=300)  
    
        #Hospitalizations
        font = 24
        fig, ax = plt.subplots() 
        ax.plot(t_star, H_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
        ax.plot(t_star, H/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('Current hospitalized ($H$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Current_hospitalized.png', dpi=300)  
    
        #Death 
        font = 24
        fig, ax = plt.subplots() 
        ax.plot(t_star, D/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('Current death ($D$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Current_death.png', dpi=300)  
        
        #New cases 
        #Confirmed Cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I_new_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
        # ax.plot(t_star, I_new_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
        ax.plot(t_star[1:], I_new/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('New cases ($I$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'new_cases.png', dpi=300)  
    
        # #Hospitalized Cases
        # font = 24
        # fig, ax = plt.subplots()
        # ax.plot(t_star[1:], H_new/sf, 'r-',lw=2,label='PINN') 
        # # ax.set_xlim(0-0.5,180)
        # # ax.set_ylim(0-0.5,6000+0.5)
        # ax.legend(fontsize=22)
        # ax.tick_params(axis='both', labelsize = 24)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        # ax.grid(True)
        # ax.set_xlabel('Days', fontsize = font)
        # ax.set_ylabel('New hospitalized ($H$)', fontsize = font) 
        # fig.set_size_inches(w=13,h=6.5)
        # if SAVE_FIG:
        #     plt.savefig(save_results_to +'new_hospitalized.png', dpi=300)  
    
        #Death Cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, D_new_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
        # ax.plot(t_star, D_new_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
        ax.plot(t_star[1:], D_new/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('New death ($D$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'new_death.png', dpi=300)  
        
        #Accumulative confirmed Cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I_sum_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
        # ax.plot(t_star, I_sum_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
        ax.plot(t_star, I_sum/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('Cumulative cases ($I_{sum}$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Cumulative_cases.png', dpi=300)  
    
        #Accumulative hospitalized Cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, H_sum/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('Cumulative hospitalized ($H_{sum}$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Cumulative_hospitalized.png', dpi=300)  
    
        #Accumulative death cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, D_sum_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
        # ax.plot(t_star, D_sum_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
        ax.plot(t_star, D_sum/sf, 'r-',lw=2,label='PINN') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('Cumulative death ($D_{sum}$)', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Cumulative_death.png', dpi=300) 
    
        #BetaI curve 
        font = 24
        fig, ax = plt.subplots()  
        ax.plot(t_star, BetaI, 'r-',lw=2) 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('$Beta_{I}$', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'BetaI.png', dpi=300)  
       
        #RC curve 
        font = 24
        fig, ax = plt.subplots()  
        ax.plot(t_star, Rc, 'r-',lw=2) 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('$R_{c}$', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Rc.png', dpi=300)
    
        #p curve 
        font = 24
        fig, ax = plt.subplots()  
        ax.plot(t_star, p, 'r-', lw=2, label='PINN')    
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('$p$', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'p.png', dpi=300)
           
        #q curve 
        font = 24
        fig, ax = plt.subplots()  
        ax.plot(t_star, q, 'r-', lw=2, label='PINN')    
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('$q$', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'q.png', dpi=300)
          
        # Current SEPIJQDHR
        font = 24
        fig, ax = plt.subplots() 
        ax.plot(t_star, S/sf, lw=2,label='S')  
        ax.plot(t_star, E/sf, lw=2,label='E')  
        ax.plot(t_star, PreS/sf, lw=2,label='P')  
        ax.plot(t_star, I/sf, lw=2,label='I')  
        ax.plot(t_star, J/sf, lw=2,label='J')  
        ax.plot(t_star, Qua/sf, lw=2,label='Q')  
        ax.plot(t_star, D/sf, lw=2,label='D')  
        ax.plot(t_star, H/sf, lw=2,label='H')  
        ax.plot(t_star, R/sf, lw=2,label='R') 
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize = 24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.grid(True)
        ax.set_xlabel('Days', fontsize = font)
        ax.set_ylabel('Current Values', fontsize = font) 
        fig.set_size_inches(w=13,h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to +'Current.png', dpi=300) 
    
 