# -*- coding: utf-8 -*-
""" 

Solve the inverse problem of the coupled ODE system 
#dS/dt = - BetaI(t)*I/N * S
#S = N-E-I-J-D-H-R 
dI/dt  = BetaI(t)*I/N * S - gamma * I 
dH/dt  = (p*gamma) * I - (q*phiD) * H - ((1-q)*phiR) * H 
dD/dt  = (q*phiD) * H 
dR/dt  = ((1-p)*gamma) * I + ((1-q)*phiR) * H 
dI_sum/dt  = BetaI(t)*I/N * S
dH_sum/dt  = (p*gamma) * I
 
Here the parameters BetaI, p, and q are inferred as time-dependent function
gamma, phiD, and phiR are given 
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
# from numpy import matlib as mb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
class PhysicsInformedNN:
    #Initialize the class
    def __init__(self, t_train, I_new_train, D_new_train, H_new_train, 
                 I_sum_train, D_sum_train, H_sum_train, U0, t_f, lb, ub, N, 
                 layers, layers_kappa, layers_BetaI, layers_p, layers_q): 
        
        self.N = N  
        
        #Data for training  
        self.t_train = t_train  
        self.I_new_train = I_new_train 
        self.D_new_train = D_new_train  
        self.H_new_train = H_new_train
        self.I_sum_train = I_sum_train 
        self.D_sum_train = D_sum_train  
        self.H_sum_train = H_sum_train
        self.I0_new = U0[0] 
        self.D0_new = U0[1] 
        self.H0_new = U0[2]
        self.I0_sum = U0[3] 
        self.D0_sum = U0[4] 
        self.H0_sum = U0[5]  
        self.t_f = t_f  
        
        #Bounds 
        self.lb = lb
        self.ub = ub

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)     
        self.weights_BetaI, self.biases_BetaI = self.initialize_NN(layers_BetaI)    
        self.weights_p, self.biases_p = self.initialize_NN(layers_p)    
        self.weights_q, self.biases_q = self.initialize_NN(layers_q)
  
        #Fixed parameters
        self.N = N 
        self.gamma = tf.Variable(1.0/6.0,dtype=tf.float64,trainable=False)  
        self.phiD = tf.Variable(1.0/15.0,dtype=tf.float64,trainable=False) 
        self.phiR = tf.Variable(1.0/7.5,dtype=tf.float64,trainable=False)  

        #tf placeholders and graph
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
        self.I0_new_u = tf.placeholder(tf.float64, shape=[None, self.I0_new.shape[1]])
        self.D0_new_u = tf.placeholder(tf.float64, shape=[None, self.D0_new.shape[1]]) 
        self.H0_new_u = tf.placeholder(tf.float64, shape=[None, self.H0_new.shape[1]])  
        self.I0_sum_u = tf.placeholder(tf.float64, shape=[None, self.I0_sum.shape[1]])
        self.D0_sum_u = tf.placeholder(tf.float64, shape=[None, self.D0_sum.shape[1]]) 
        self.H0_sum_u = tf.placeholder(tf.float64, shape=[None, self.H0_sum.shape[1]]) 
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])   

        # physics informed neural networks
        self.S_pred, self.I_pred, self.H_pred, self.D_pred, self.R_pred, self.I_sum_pred, self.H_sum_pred = self.net_u(self.t_u)  
        self.D_sum_pred = self.D_pred
        
        self.BetaI_pred = self.net_BetaI(self.t_u)
        self.p_pred = self.net_p (self.t_u)
        self.q_pred = self.net_q (self.t_u) 
        
        self.I_new_pred = self.I_sum_pred[1:,:] - self.I_sum_pred[:-1,:]
        self.D_new_pred = self.D_sum_pred[1:,:] - self.D_sum_pred[:-1,:]
        self.H_new_pred = self.H_sum_pred[1:,:] - self.H_sum_pred[:-1,:] 
        
        self.I0_new_pred = self.I_new_pred[0] 
        self.D0_new_pred = self.D_new_pred[0]
        self.H0_new_pred = self.H_new_pred[0]
        self.I0_sum_pred = self.I_sum_pred[0] 
        self.D0_sum_pred = self.D_sum_pred[0]
        self.H0_sum_pred = self.H_sum_pred[0]   
        
        self.I_f, self.H_f, self.D_f, self.R_f, self.I_sum_f, self.H_sum_f = self.net_f(self.t_tf)
        
        # loss
        self.lossU0 = tf.reduce_mean(tf.square(self.I0_new_u - self.I0_new_pred)) + \
            tf.reduce_mean(tf.square(self.D0_new_u - self.D0_new_pred)) + \
            tf.reduce_mean(tf.square(self.H0_new_u - self.H0_new_pred)) + \
            tf.reduce_mean(tf.square(self.I0_sum_u - self.I0_sum_pred)) + \
            tf.reduce_mean(tf.square(self.D0_sum_u - self.D0_sum_pred)) + \
            tf.reduce_mean(tf.square(self.H0_sum_u - self.H0_sum_pred)) 
        
        self.lossU = tf.reduce_mean(tf.square(self.I_new_u[:-1,:] - self.I_new_pred)) + \
            tf.reduce_mean(tf.square(self.D_new_u[:-1,:] - self.D_new_pred)) + \
            tf.reduce_mean(tf.square(self.H_new_u[:-1,:] - self.H_new_pred)) + \
            tf.reduce_mean(tf.square(self.I_sum_u - self.I_sum_pred)) + \
            tf.reduce_mean(tf.square(self.D_sum_u - self.D_sum_pred)) + \
            tf.reduce_mean(tf.square(self.H_sum_u - self.H_sum_pred)) 
        
        self.lossF = tf.reduce_mean(tf.square(self.I_f)) + tf.reduce_mean(tf.square(self.H_f)) + \
            tf.reduce_mean(tf.square(self.D_f)) + tf.reduce_mean(tf.square(self.R_f)) + \
            tf.reduce_mean(tf.square(self.I_sum_f)) + tf.reduce_mean(tf.square(self.H_sum_f)) 

        self.loss = self.lossU0 + self.lossU + self.lossF  


        #Optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
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
    
    def net_BetaI(self, t):
        BetaI = self.neural_net(t, self.weights_BetaI, self.biases_BetaI)
        bound_b = [tf.constant(0.05, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64)] 
        return bound_b[0]+(bound_b[1]-bound_b[0])*tf.sigmoid(BetaI)
    
    def net_p(self, t):
        p = self.neural_net(t, self.weights_p, self.biases_p)
        return tf.sigmoid(p)
        # p = 0.5*(1-tf.tanh(t-50.0))*(0.5-0.1)+0.1
        # return p

    
    def net_q(self, t):
        # q = self.neural_net(t, self.weights_q, self.biases_q) 
        # return tf.sigmoid(q)
        q = 0.5*(1-tf.tanh(t-150.0))*(0.5-0.2)+0.2
        return q

    def net_u(self, t):
        SIHDR = self.neural_net(t, self.weights, self.biases)
        SIHDR = SIHDR**2
        I = SIHDR[:,0:1] 
        H = SIHDR[:,1:2]
        D = SIHDR[:,2:3]
        R = SIHDR[:,3:4]
        I_sum = SIHDR[:,4:5]
        H_sum = SIHDR[:,5:6]    
        S = self.N-I-H-D-R  
        return S, I, H, D, R, I_sum, H_sum
    
    
    def net_f(self, t):
        #load fixed parameters 
        gamma = self.gamma  
        phiD = self.phiD
        phiR = self.phiR  
        
        #load time-dependent parameters
        betaI = self.net_BetaI(t)  
        p = self.net_p(t)
        q = self.net_q(t)   
        
        #Obtain SIHDR from Neural network 
        S, I, H, D, R, I_sum, H_sum = self.net_u(t) 
        
        I_t = tf.gradients(I, t, unconnected_gradients='zero')[0]
        H_t = tf.gradients(H, t, unconnected_gradients='zero')[0]
        D_t = tf.gradients(D, t, unconnected_gradients='zero')[0]
        R_t = tf.gradients(R, t, unconnected_gradients='zero')[0]
        I_sum_t = tf.gradients(I_sum, t, unconnected_gradients='zero')[0]
        H_sum_t = tf.gradients(H_sum, t, unconnected_gradients='zero')[0]

        
        #Residuals  
        f_I = I_t - betaI*I/N * S + gamma * I
        f_H = H_t - (p*gamma) * I + (q*phiD) * H + ((1-q)*phiR) * H 
        f_D = D_t - (q*phiD)*H
        f_R = R_t - ((1-p)*gamma) * I - ((1-q)*phiR) * H 
        f_I_sum = I_sum_t - betaI*I/N * S
        f_H_sum = H_sum_t - (p*gamma)*I 

        return f_I, f_H, f_D, f_R, f_I_sum, f_H_sum  

    def callback(self, loss, lossU0, lossU, lossF):
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF]))
        print('Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e' % (loss, lossU0, lossU, lossF))        
   
    def train(self, nIter):  
                
        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f, 
                   self.I_new_u: self.I_new_train, self.D_new_u: self.D_new_train, self.H_new_u: self.H_new_train, 
                   self.I_sum_u: self.I_sum_train, self.H_sum_u: self.H_sum_train, self.D_sum_u: self.D_sum_train,  
                   self.I0_new_u: self.I0_new, self.D0_new_u: self.D0_new, self.H0_new_u: self.H0_new, 
                   self.I0_sum_u: self.I0_sum, self.D0_sum_u: self.D0_sum, self.H0_sum_u: self.H0_sum}
        
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
                total_records.append(np.array([it, loss_value, lossU0_value, lossU_value, lossF_value]))
                print('It: %d, Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e, Time: %.2f' % 
                      (it, loss_value, lossU0_value, lossU_value, lossF_value, elapsed))
                start_time = time.time()
                
        
        if LBFGS:
            self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict, #Inputs of the minimize operator 
                                fetches = [self.loss, self.lossU0, self.lossU, self.lossF], 
                                loss_callback = self.callback) #Show the results of minimize operator

    def predict(self, t_star):        

        tf_dict = {self.t_u: t_star, self.I0_new_u: self.I0_new, self.D0_new_u: self.D0_new, self.H0_new_u: self.H0_new} 
        
        S = self. sess.run(self.S_pred, tf_dict) 
        I = self. sess.run(self.I_pred, tf_dict) 
        H = self. sess.run(self.H_pred, tf_dict)
        D = self. sess.run(self.D_pred, tf_dict)
        R = self. sess.run(self.R_pred, tf_dict)
        
        I_sum = self. sess.run(self.I_sum_pred, tf_dict)
        H_sum = self. sess.run(self.H_sum_pred, tf_dict) 
        D_sum = self. sess.run(self.D_sum_pred, tf_dict) 
        
        I_new = self. sess.run(self.I_new_pred, tf_dict)
        D_new = self. sess.run(self.D_new_pred, tf_dict)
        H_new = self. sess.run(self.H_new_pred, tf_dict) 
        
        BetaI = self.sess.run(self.BetaI_pred, tf_dict) 
        p = self.sess.run(self.p_pred, tf_dict) 
        q = self.sess.run(self.q_pred, tf_dict) 
        return S, I, H, D, R, I_new, H_new, D_new, I_sum, H_sum, D_sum, BetaI, p, q 
        
############################################################    
#%%
if __name__=="__main__":   
    
    #Architecture of of the NN 
    layers=[1] + 5*[30] + [6] 
    layers_kappa=[1] + 2*[15] + [1]
    layers_BetaI=[1] + 2*[15] + [1]
    layers_p=[1] + 2*[15] + [1]
    layers_q=[1] + 2*[15] + [1]

    #Load data
    data_frame = pandas.read_csv('../Data/data-by-day.csv')  
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
    
    #Scaling 
    sf = 1e-4
    N = N * sf 
    I_new_star = I_new_star * sf 
    D_new_star = D_new_star * sf
    H_new_star = H_new_star * sf  
    I_sum_star = I_sum_star * sf 
    H_sum_star = H_sum_star * sf 
    D_sum_star = D_sum_star * sf  
    
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
    U0 = [I0_new, D0_new, H0_new, I0_sum, D0_sum, H0_sum] 
    
    #Residual points  
    N_f = 300 #5 * len(t_train)
    # t_f = np.linspace(lb, ub, num = N_f) #uniformed grid for evaluating fractional derivative  
    t_f = lb + (ub-lb)*lhs(1,N_f)    
#%%
    ######################################################################
    ######################## Training and Predicting ###############################
    ###################################################################### 
    # t_train = (t_star-lb)/(ub-lb)  
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
    for j in range(5):
        casenumber = 'set' + str(j+1)

        relative_path_results = '/Model-SIHDR/Train-Results-'+dt_string+'-'+casenumber+'/'
        save_results_to = current_directory + relative_path_results
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        relative_path = '/Model-SIHDR/Train-model-'+dt_string+'-'+casenumber+'/' 
        save_models_to = current_directory + relative_path
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)
            break

    #Training 
    total_records = []
    total_records_LBFGS = []
    model = PhysicsInformedNN(t_train, I_new_train, D_new_train, H_new_train, 
                  I_sum_train, D_sum_train, H_sum_train, U0, t_f, lb, ub, N, 
                  layers,  layers_kappa, layers_BetaI, layers_p, layers_q)

    # LBFGS = True 
    LBFGS = False
    model.train(1000) #Training with n iterations 

    model.saver.save(model.sess, save_models_to+"model.ckpt")

    #Predicting   
    S, I, H, D, R, I_new, H_new, D_new, I_sum, H_sum, D_sum, BetaI, p, q = model.predict(t_star) 
    import datetime
    end_time = time.time()
    print(datetime.timedelta(seconds=int(end_time-start_time)))
     
    #Calculate RC  
    Rc = BetaI /(1.0/6.0)
    
#%%    
    #save data
    np.savetxt(save_results_to + 'S.txt', S.reshape((-1,1))) 
    np.savetxt(save_results_to + 'I.txt', I.reshape((-1,1))) 
    np.savetxt(save_results_to + 'D.txt', D.reshape((-1,1)))
    np.savetxt(save_results_to + 'H.txt', H.reshape((-1,1)))
    np.savetxt(save_results_to + 'R.txt', R.reshape((-1,1)))
    np.savetxt(save_results_to + 'I_new.txt', I_new.reshape((-1,1)))
    np.savetxt(save_results_to + 'D_new.txt', D_new.reshape((-1,1)))
    np.savetxt(save_results_to + 'H_new.txt', H_new.reshape((-1,1)))
    np.savetxt(save_results_to + 'I_sum.txt', I_sum.reshape((-1,1))) 
    np.savetxt(save_results_to + 'H_sum.txt', H_sum.reshape((-1,1))) 
    np.savetxt(save_results_to + 'D_sum.txt', D_sum.reshape((-1,1))) 
    #save BetaI, Rc, and sigma
    np.savetxt(save_results_to + 't_star.txt', t_star.reshape((-1,1))) 
    np.savetxt(save_results_to + 'Rc.txt', Rc.reshape((-1,1)))   
    np.savetxt(save_results_to + 'BetaI.txt', BetaI.reshape((-1,1)))
    np.savetxt(save_results_to + 'p.txt', p.reshape((-1,1)))
    np.savetxt(save_results_to + 'q.txt', q.reshape((-1,1)))


#%% 
    #records for Adam
    N_Iter = len(total_records)
    iteration = np.asarray(total_records)[:,0]
    loss_his = np.asarray(total_records)[:,1]
    loss_his_u0  = np.asarray(total_records)[:,2]
    loss_his_u  = np.asarray(total_records)[:,3]
    loss_his_f  = np.asarray(total_records)[:,4]  
    
    #records for LBFGS
    if LBFGS:
        N_Iter_LBFGS = len(total_records_LBFGS)
        iteration_LBFGS = np.arange(N_Iter_LBFGS)+N_Iter*100
        loss_his_LBFGS = np.asarray(total_records_LBFGS)[:,0]
        loss_his_u0_LBFGS = np.asarray(total_records_LBFGS)[:,1]
        loss_his_u_LBFGS  = np.asarray(total_records_LBFGS)[:,2]
        loss_his_f_LBFGS  = np.asarray(total_records_LBFGS)[:,3]  

#%%    
    #save records
    np.savetxt(save_results_to + 'iteration.txt', iteration.reshape((-1,1))) 
    np.savetxt(save_results_to + 'loss_his.txt', loss_his.reshape((-1,1)))
    np.savetxt(save_results_to + 'loss_his_u0.txt', loss_his_u0.reshape((-1,1))) 
    np.savetxt(save_results_to + 'loss_his_u.txt', loss_his_u.reshape((-1,1))) 
    np.savetxt(save_results_to + 'loss_his_f.txt', loss_his_f.reshape((-1,1)))   

    if LBFGS:    
        np.savetxt(save_results_to + 'iteration_LBFGS.txt', iteration_LBFGS.reshape((-1,1))) 
        np.savetxt(save_results_to + 'loss_his_LBFGS.txt', loss_his_LBFGS.reshape((-1,1)))
        np.savetxt(save_results_to + 'loss_his_u0_LBFGS.txt', loss_his_u0_LBFGS.reshape((-1,1))) 
        np.savetxt(save_results_to + 'loss_his_u_LBFGS.txt', loss_his_u_LBFGS.reshape((-1,1))) 
        np.savetxt(save_results_to + 'loss_his_f_LBFGS.txt', loss_his_f_LBFGS.reshape((-1,1)))    

#%%    
    #History of loss 
    fig, ax = plt.subplots() 
    plt.yscale('log')  
    plt.plot(iteration, loss_his, 'k-', lw = 4, label='$loss$')
    plt.plot(iteration, loss_his_u0, 'r-', lw = 4, label='$loss_{u0}$')
    plt.plot(iteration, loss_his_u, 'b-', lw = 4, label='$loss_u$')
    plt.plot(iteration, loss_his_f, 'c-', lw = 4, label='$loss_f$')  
    if LBFGS:
        plt.plot(iteration_LBFGS, loss_his_LBFGS, 'k-', lw = 4)
        plt.plot(iteration_LBFGS, loss_his_u0_LBFGS, 'r-', lw = 4)
        plt.plot(iteration_LBFGS, loss_his_u_LBFGS, 'b-', lw = 4)
        plt.plot(iteration_LBFGS, loss_his_f_LBFGS, 'c-', lw = 4) 
    ax.legend(fontsize=50, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50) 
    plt.rc('font', size=60)
    ax.grid(True)
    ax.set_xlabel('Iteration', fontsize = 50) 
    ax.set_ylabel('loss values', fontsize = 70) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'History_loss.png', dpi=300) 
    plt.savefig(save_results_to + 'History_loss.pdf', dpi=300) 
    
#%%
    ######################################################################
    ############################# Plotting ###############################
    ###################################################################### 
#%% 
    # date_total = np.arange('2020-03-08', '2020-11-13', dtype='datetime64[D]')[:,None] 
    date_total = t_train

#%%
    #Current Suspectious 
    fig, ax = plt.subplots() 
    ax.plot(date_total, S/sf, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$S$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Current_Suspectious.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Current_Suspectious.png', dpi=300) 

#%%
    #Current Infectious 
    fig, ax = plt.subplots() 
    ax.plot(date_total, I/sf, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$I$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Current_Infectious.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Current_Infectious.png', dpi=300) 

#%%
    #Current Hospitalized   
    fig, ax = plt.subplots() 
    ax.plot(date_total, H/sf, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$H$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Current_Hosppitalized.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Current_Hosppitalized.png', dpi=300) 

#%%
    #Current death   
    fig, ax = plt.subplots() 
    ax.plot(date_total, D/sf, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$D$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Current_Death.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Current_Death.png', dpi=300) 

#%%
    #Current Removed 
    fig, ax = plt.subplots() 
    ax.plot(date_total, R/sf, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$R$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Current_Removed.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Current_Removed.png', dpi=300) 

#%%
    #New infectious
    fig, ax = plt.subplots() 
    ax.plot(date_total, I_new_train/sf, 'ro', lw=5, markersize=10, label='Data-7davg')
    ax.plot(date_total[:-1], I_new/sf, 'k-', lw=5, label = 'fPINNs')  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=60, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$I^{new}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'New_Infectious.pdf', dpi=300) 
    plt.savefig(save_results_to + 'New_Infectious.png', dpi=300) 

#%%
    #Cumulative Infectious 
    fig, ax = plt.subplots() 
    ax.plot(date_total, I_sum_train/sf, 'ro', lw=5, markersize=10, label='Data-7davg')
    ax.plot(date_total, I_sum/sf, 'k-', lw=5, label = 'fPINNs')  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=60, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$I^{c}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Cumulative_Infectious.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Cumulative_Infectious.png', dpi=300) 

#%%
    #New Hosptalized
    fig, ax = plt.subplots() 
    ax.plot(date_total, H_new_train/sf, 'ro', lw=5, markersize=10, label='Data-7davg')
    ax.plot(date_total[:-1], H_new/sf, 'k-', lw=5, label = 'fPINNs')  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=60, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$H^{new}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'New_Hosptalized.pdf', dpi=300) 
    plt.savefig(save_results_to + 'New_Hosptalized.png', dpi=300) 

#%%
    #Cumulative Hosptalized 
    fig, ax = plt.subplots() 
    ax.plot(date_total, H_sum_train/sf, 'ro', lw=5, markersize=10, label='Data-7davg')
    ax.plot(date_total, H_sum/sf, 'k-', lw=5, label = 'fPINNs')  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=60, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$H^{c}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Cumulative_Hosptalized.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Cumulative_Hosptalized.png', dpi=300) 


#%%
    #New death
    fig, ax = plt.subplots() 
    ax.plot(date_total, D_new_train/sf, 'ro', lw=5, markersize=10, label='Data-7davg')
    ax.plot(date_total[:-1], D_new/sf, 'k-', lw=5, label = 'fPINNs')  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=60, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$D^{new}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'New_Death.pdf', dpi=300) 
    plt.savefig(save_results_to + 'New_Death.png', dpi=300) 

#%%
    #Cumulative Death 
    fig, ax = plt.subplots() 
    ax.plot(date_total, D_sum_train/sf, 'ro', lw=5, markersize=10, label='Data-7davg')
    ax.plot(date_total, D_sum/sf, 'k-', lw=5, label = 'fPINNs')  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=60, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$D^{c}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Cumulative_Death.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Cumulative_Death.png', dpi=300) 

#%%
    #BetaI
    fig, ax = plt.subplots() 
    ax.plot(date_total, BetaI, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50) 
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$Beta_{I}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'BetaI.pdf', dpi=300) 
    plt.savefig(save_results_to + 'BetaI.png', dpi=300)  

#%%
    #Rc
    fig, ax = plt.subplots() 
    ax.plot(date_total, Rc, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50) 
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$R_{c}$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'Rc.pdf', dpi=300) 
    plt.savefig(save_results_to + 'Rc.png', dpi=300)  

#%%
    #p
    fig, ax = plt.subplots() 
    ax.plot(date_total, p, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50) 
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$p$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'p.pdf', dpi=300) 
    plt.savefig(save_results_to + 'p.png', dpi=300)  

#%%
    #q
    fig, ax = plt.subplots() 
    ax.plot(date_total, q, 'k-', lw=5)  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    # ax.legend(fontsize=18, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 50)
    ax.tick_params(axis='y', labelsize = 50) 
    plt.rc('font', size=60)
    ax.grid(True)
    # ax.set_xlabel('Date', fontsize = font) 
    ax.set_ylabel('$q$', fontsize = 80) 
    fig.set_size_inches(w=25, h=12.5)
    plt.savefig(save_results_to + 'q.pdf', dpi=300) 
    plt.savefig(save_results_to + 'q.png', dpi=300)  



