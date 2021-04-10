# -*- coding: utf-8 -*-
""" 
@Title:
    Identifiability and predictability of
    integer- and fractional-order epidemiological models
    using physics-informed neural networks
@author: 
    Ehsan Kharazmi & Min Cai
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
from scipy.special import jacobi

start_time = time.time()


# np.random.seed(1234)
# tf.set_random_seed(1234)
# tf.random.set_seed(1234)

#%%
class PhysicsInformedNN:
    #Initialize the class
    def __init__(self, t_f, t_train, I_train, R_train, D_train, 
                 U0, lb, ub, N, layers, layers_Beta): 


        self.N = N  
        
        #Data for training 
        self.t_f = t_f    
        self.t_train = t_train 
        self.I_train = I_train  
        self.R_train = R_train 
        self.D_train = D_train 
        self.S0 = U0[0] 
        self.I0 = U0[1] 
        self.R0 = U0[2]
        self.D0 = U0[3]  
        
        #Time division s
        self.M = len(t_f)-1
        self.tau = t_f[1]-t_f[0] 

        #Bounds 
        self.lb = lb
        self.ub = ub

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  
        self.weights_Beta, self.biases_Beta = self.initialize_NN(layers_Beta)         

        self.Kappa1_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64) , dtype=tf.float64, trainable=True)
        self.Kappa2_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64) , dtype=tf.float64, trainable=True)
        self.Kappa3_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64) , dtype=tf.float64, trainable=True)
        self.Kappa4_COEF = tf.Variable(tf.zeros([poly_order,1], dtype=tf.float64) , dtype=tf.float64, trainable=True)
        
        #Fixed parameters
        self.N = N 
        self.a = tf.Variable(2.15e-2,dtype=tf.float64,trainable=False)  
        self.b = tf.Variable(0.48e-2,dtype=tf.float64,trainable=False)  
        
        #tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.saver = tf.train.Saver()
        

        # placeholders for inputs  
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]]) 
        self.t_u = tf.placeholder(tf.float64, shape=[None, self.t_train.shape[1]])  
        self.I_u = tf.placeholder(tf.float64, shape=[None, self.I_train.shape[1]])  
        self.R_u = tf.placeholder(tf.float64, shape=[None, self.R_train.shape[1]])  
        self.D_u = tf.placeholder(tf.float64, shape=[None, self.D_train.shape[1]])    
        self.S0_u = tf.placeholder(tf.float64, shape=[None, self.S0.shape[1]])  
        self.I0_u = tf.placeholder(tf.float64, shape=[None, self.I0.shape[1]])  
        self.R0_u = tf.placeholder(tf.float64, shape=[None, self.R0.shape[1]])  
        self.D0_u = tf.placeholder(tf.float64, shape=[None, self.D0.shape[1]])   

        # physics informed neural networks
        self.S_pred, self.I_pred, self.R_pred, self.D_pred = self.net_u(self.t_u) 
        
        self.BetaI = self.net_Beta(self.t_u)

        self.Kappa_pred1 = self.net_Kappa1_plot()
        self.Kappa_pred2 = self.net_Kappa2_plot()
        self.Kappa_pred3 = self.net_Kappa3_plot()
        self.Kappa_pred4 = self.net_Kappa4_plot()  
        
        self.S0_pred = self.S_pred[0] 
        self.I0_pred = self.I_pred[0]
        self.R0_pred = self.R_pred[0]
        self.D0_pred = self.D_pred[0]    
        
        self.S_f, self.I_f, self.R_f, self.D_f, self.R_con = self.net_f(self.t_tf)
        
        # loss
        self.lossU0 = tf.reduce_mean(tf.square(self.I0_u - self.I0_pred)) + \
            tf.reduce_mean(tf.square(self.R0_u - self.R0_pred)) + \
            tf.reduce_mean(tf.square(self.D0_u - self.D0_pred))
            # tf.reduce_mean(tf.square(self.S0_u - self.S0_pred)) 
        
        self.lossU = 8*tf.reduce_mean(tf.square(self.I_pred - self.I_u)) + \
            tf.reduce_mean(tf.square(self.R_pred - self.R_u)) + \
            60*tf.reduce_mean(tf.square(self.D_pred - self.D_u))
        
        self.lossF =tf.reduce_mean(tf.square(self.S_f))\
                    + tf.reduce_mean(tf.square(self.I_f))\
                    + tf.reduce_mean(tf.square(self.D_f))\
                    + tf.reduce_mean(tf.square(self.R_f))\
                    + tf.reduce_mean(tf.square(self.R_con))

        self.loss = 1*self.lossU0 + 5*self.lossU + self.lossF  


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

    def net_Beta(self, t):
        BetaI = self.neural_net(t, self.weights_Beta, self.biases_Beta)
        bound_b = [tf.constant(0.0, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64)] 
        return bound_b[0]+(bound_b[1]-bound_b[0])*tf.sigmoid(BetaI)
    
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
        SIDR = self.neural_net(t, self.weights, self.biases)
        # SIDR = SIDR**2
        S = SIDR[:,0:1] 
        I = SIDR[:,1:2] 
        R = SIDR[:,2:3]
        D = SIDR[:,3:4]   
        return S, I, R, D 

    def net_Kappa1(self):
        polys = tf.constant(np.transpose(Jacobi_polys[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa1_COEF)
        # return tf.sigmoid(Kappa)
        return 0.2 + 0.8 * tf.sigmoid(Kappa)
    def net_Kappa2(self):
        polys = tf.constant(np.transpose(Jacobi_polys[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa2_COEF)
        # return tf.sigmoid(Kappa)
        return 0.2 + 0.8 * tf.sigmoid(Kappa)
    def net_Kappa3(self):
        polys = tf.constant(np.transpose(Jacobi_polys[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa3_COEF)
        # return tf.sigmoid(Kappa)
        return 0.2 + 0.8 * tf.sigmoid(Kappa)
    def net_Kappa4(self):
        polys = tf.constant(np.transpose(Jacobi_polys[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa4_COEF)
        # return tf.sigmoid(Kappa) 
        return 0.2 + 0.8 * tf.sigmoid(Kappa)

    def net_Kappa1_plot(self):
        polys = tf.constant(np.transpose(Jacobi_polys_plots[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa1_COEF)
        # return tf.sigmoid(Kappa)
        return 0.2 + 0.8 * tf.sigmoid(Kappa)
    def net_Kappa2_plot(self):
        polys = tf.constant(np.transpose(Jacobi_polys_plots[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa2_COEF)
        # return tf.sigmoid(Kappa)
        return 0.2 + 0.8 * tf.sigmoid(Kappa)
    def net_Kappa3_plot(self):
        polys = tf.constant(np.transpose(Jacobi_polys_plots[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa3_COEF)
        # return tf.sigmoid(Kappa)
        return 0.2 + 0.8 * tf.sigmoid(Kappa)
    def net_Kappa4_plot(self):
        polys = tf.constant(np.transpose(Jacobi_polys_plots[:poly_order,:]), dtype=tf.float64)  
        Kappa = tf.matmul(polys, self.Kappa4_COEF)
        # return tf.sigmoid(Kappa) 
        return 0.2 + 0.8 * tf.sigmoid(Kappa)
    
    #fractional differential coefficients for the L1 approximation    
    def FDM1(self, Kappa):
        m = self.M  #int
        Tau = self.tau #array
        kappa_vec = tf.reshape(Kappa, [m+1,1]) #tnsor
        kappa_mat = tf.tile(kappa_vec, [1, m-1])
        
        idx = np.tril_indices(m+1, k=-1) 
        Temp1 = np.zeros([m+1,m+1])  
        Temp1[idx] = idx[0]-idx[1] 
        Temp1 = np.tril(Temp1, k=-2) #(m+1,m+1) numpy array
        Temp1 = tf.constant(Temp1, dtype = tf.float64)  #(m+1,m+1) tensor
        Temp2 = -np.eye(m+1) 
        Temp2[idx] = idx[0]-idx[1]-1
        Temp2 = np.tril(Temp2, k=-2) 
        Temp2 = tf.constant(Temp2, dtype = tf.float64)
        Temp3 = -2*np.eye(m+1)  
        Temp3[idx] = idx[0]-idx[1]-2
        Temp3 = np.tril(Temp3, k=-2)
        Temp3 = tf.constant(Temp3, dtype = tf.float64) 
        A = np.concatenate((np.zeros((1,m)), np.eye(m)), axis=0, out=None)
        A = tf.constant(A[:,0:m-1], dtype = tf.float64) 
        Temp = tf.pow(Temp1[:,0:m-1],1.0-kappa_mat) -\
            2*tf.pow(Temp2[:,0:m-1],1.0-kappa_mat) +\
                tf.pow(Temp3[:,0:m-1],1.0-kappa_mat) + A
                
        L_Temp1 = tf.constant(np.arange(m), dtype = tf.float64) #np.arange(m) 
        L_Temp1 = tf.pow(tf.reshape(L_Temp1, [m,1]), 1.0-kappa_vec[1:m+1, 0:1])
        L_Temp2 = tf.constant(np.arange(m)+1, dtype = tf.float64) #np.arange(m) + 1 
        L_Temp2 = tf.pow(tf.reshape(L_Temp2, [m,1]), 1.0-kappa_vec[1:m+1, 0:1])
        L_Temp = tf.concat((tf.zeros((1,1), dtype = tf.float64), L_Temp1-L_Temp2), axis=0)
        
        R_Temp = tf.concat((tf.zeros((m,1), dtype = tf.float64), tf.ones((1,1), dtype = tf.float64)), axis=0)
        
        coeff_mat = tf.concat((L_Temp, Temp, R_Temp), axis=1)
        
        c = tf.tile(tf.math.divide(tf.pow(Tau, -kappa_vec), tf.exp(tf.lgamma(2-kappa_vec))), tf.constant([1, m+1], dtype = tf.int32))
        
        coeff_mat = tf.multiply(c, coeff_mat) 
        return coeff_mat
    
    def net_f(self, t): 
        
        #load time-dependent parameters
        # r0 = 1.2e-6 
        # r = 0.66*r0*tf.exp(-0.5*t) + 0.34*r0 
        r = self.net_Beta(t)
        
        #Obtain SIHDR from Neural network 
        S, I, R, D = self.net_u(t) 
        
        #Time derivatives 
        #Fractional differential matrix  
        Kappa1 = self.net_Kappa1()
        Kappa2 = self.net_Kappa2()
        Kappa3 = self.net_Kappa3()
        Kappa4 = self.net_Kappa4() 

        DiffMat1 = self.FDM1(Kappa1) 
        DiffMat2 = self.FDM1(Kappa2) 
        DiffMat3 = self.FDM1(Kappa3) 
        DiffMat4 = self.FDM1(Kappa4)  

        #fractional time derivatives 
        # DM = self.DM  
        S_t = tf.matmul(DiffMat1, S) 
        I_t = tf.matmul(DiffMat2, I) 
        R_t = tf.matmul(DiffMat3, R)
        D_t = tf.matmul(DiffMat4, D)   
        
        T = tf.constant(7.0, dtype = tf.float64)  
        
        ## fractional derivative
        S_t = tf.pow(T, Kappa1-1)*S_t/tf.exp(tf.lgamma(1.0+Kappa1))
        I_t = tf.pow(T, Kappa2-1)*I_t/tf.exp(tf.lgamma(1.0+Kappa2))
        R_t = tf.pow(T, Kappa3-1)*R_t/tf.exp(tf.lgamma(1.0+Kappa3))
        D_t = tf.pow(T, Kappa4-1)*D_t/tf.exp(tf.lgamma(1.0+Kappa4))          
        
        #Residuals  
        f_S = S_t + r * S * I 
        f_I = I_t - r * S * I + (self.a + self.b) * I  
        f_R = R_t - self.a * I  
        f_D = D_t - self.b * I 
        f_con = S + I + R + D - self.N

        return f_S, f_I, f_R, f_D, f_con

    def callback(self, loss, lossU0, lossU, lossF):
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF]))
        print('Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e' % (loss, lossU0, lossU, lossF))        
   
    def train(self, nIter):  
                
        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f, 
                   self.I_u: self.I_train, self.R_u: self.R_train, self.D_u: self.D_train, 
                   self.S0_u: self.S0, self.I0_u: self.I0, self.R0_u: self.R0, self.D0_u: self.D0}
        
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
                Kappa1_records.append(self.sess.run(self.Kappa_pred1))
                Kappa2_records.append(self.sess.run(self.Kappa_pred2))
                Kappa3_records.append(self.sess.run(self.Kappa_pred3))
                Kappa4_records.append(self.sess.run(self.Kappa_pred4)) 
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

        tf_dict = {self.t_u: t_star}
        
        S = self. sess.run(self.S_pred, tf_dict) 
        I = self. sess.run(self.I_pred, tf_dict) 
        R = self. sess.run(self.R_pred, tf_dict)
        D = self. sess.run(self.D_pred, tf_dict) 
        
        Beta = self. sess.run(self.BetaI, tf_dict) 
        
        Kappa1 = self.sess.run(self.Kappa_pred1, tf_dict) 
        Kappa2 = self.sess.run(self.Kappa_pred2, tf_dict) 
        Kappa3 = self.sess.run(self.Kappa_pred3, tf_dict) 
        Kappa4 = self.sess.run(self.Kappa_pred4, tf_dict)  
        
        return S, I, R, D, Kappa1, Kappa2, Kappa3, Kappa4, Beta 
    
############################################################    
#%%
if __name__=="__main__":   
    
    #Architecture of of the NN 
    layers=[1] + 5*[20] + [4]   
    layers_Beta=[1] + 5*[20] + [1]   

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
    
    #Scaling 
    sf = 1e-7
    N = N * sf 
    I_star = I_star * sf 
    R_star = R_star * sf
    D_star = D_star * sf 
    
    #lower and upper bounds 
    lb = t_star.min(0)
    ub = t_star.max(0)   
    
    #Initial conditions 
    I0 = I_star[0:1,:]  
    R0 = R_star[0:1,:]  
    D0 = D_star[0:1,:]  
    S0 = N - I0 - R0 - D0  
    U0 = [S0, I0, R0, D0] 
    
    #Residual points 
    N_f = 500 #5 * len(t_star)
    t_f = np.linspace(lb, ub, num = N_f) #uniformed grid for evaluating fractional derivative 

    poly_order = 10
    t_f_mapped = -1 + 2/(ub-lb) * (t_f - lb)
    t_star_mapped = -1 + 2/(ub-lb) * (t_star - lb)
    Jacobi_polys = np.asarray([ jacobi(n,0,0)(t_f_mapped.flatten())  for n in range(0, 15)])
    Jacobi_polys_plots = np.asarray([ jacobi(n,0,0)(t_star_mapped.flatten())  for n in range(0, 15)])

    
#%%
    ######################################################################
    ######################## Training and Predicting ###############################
    ###################################################################### 
    # t_train = (t_star-lb)/(ub-lb)  
    t_train = t_star
    I_train = I_star 
    R_train = R_star 
    D_train = D_star  
    
#%%
    from datetime import datetime
    now = datetime.now()
    # dt_string = now.strftime("%m-%d-%H-%M")
    dt_string = now.strftime("%m-%d")

    #save results  
    current_directory = os.getcwd()
    for j in range(10):
        casenumber = 'set' + str(j+1)

        relative_path_results = '/SIRD-DiffKappa-Beta/Train-Results-'+dt_string+'-'+casenumber+'/'
        save_results_to = current_directory + relative_path_results
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        relative_path = '/SIRD-DiffKappa-Beta/Train-model-'+dt_string+'-'+casenumber+'/' 
        save_models_to = current_directory + relative_path
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)
            # break
        
    #%%
        #Training 
        total_records = []
        Kappa1_records = []
        Kappa2_records = []
        Kappa3_records = []
        Kappa4_records = [] 
        total_records_LBFGS = []
        model = PhysicsInformedNN(t_f, t_train, I_train, R_train, D_train, U0, lb, ub, N, layers, layers_Beta)
    
        LBFGS = True 
    #%%
        # LBFGS = False
        model.train(10000) #Training with n iterations 
    
        model.saver.save(model.sess, save_models_to+"model.ckpt")
    #%%
        #Predicting   
        S, I, R, D, Kappa1, Kappa2, Kappa3, Kappa4, Beta = model.predict(t_star) 
        import datetime
        end_time = time.time()
        print(datetime.timedelta(seconds=int(end_time-start_time)))
         
        # #Calculate RC  
        # Rc = BetaI /(1.0/6.0)
        
    #%%    
        #save data
        np.savetxt(save_results_to + 'S.txt', S.reshape((-1,1))) 
        np.savetxt(save_results_to + 'I.txt', I.reshape((-1,1))) 
        np.savetxt(save_results_to + 'R.txt', R.reshape((-1,1)))
        np.savetxt(save_results_to + 'D.txt', D.reshape((-1,1)))  
        #save BetaI, Rc, and sigma
        np.savetxt(save_results_to + 't_star.txt', t_star.reshape((-1,1)))  
        np.savetxt(save_results_to + 'Kappa1.txt', Kappa1.reshape((-1,1))) 
        np.savetxt(save_results_to + 'Kappa2.txt', Kappa2.reshape((-1,1))) 
        np.savetxt(save_results_to + 'Kappa3.txt', Kappa3.reshape((-1,1))) 
        np.savetxt(save_results_to + 'Kappa4.txt', Kappa4.reshape((-1,1)))  
        np.savetxt(save_results_to + 'Beta.txt', Beta.reshape((-1,1)))  
    
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
        ax.plot(date_total, I_star/sf, 'ro', lw=5)  
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
        #Current Removed 
        fig, ax = plt.subplots() 
        ax.plot(date_total, R_star/sf, 'ro', lw=5)  
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
        #Current death   
        fig, ax = plt.subplots() 
        ax.plot(date_total, D_star/sf, 'ro', lw=5)  
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
        #Kappa
        fig, ax = plt.subplots() 
        ax.plot(date_total, Kappa1, 'k-', lw=5, label = '$\kappa_{1}$')  
        ax.plot(date_total, Kappa2, 'r-', lw=5, label = '$\kappa_{2}$')  
        ax.plot(date_total, Kappa3, 'b-', lw=5, label = '$\kappa_{3}$')  
        ax.plot(date_total, Kappa4, 'm-', lw=5, label = '$\kappa_{4}$')  
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(0-0.5,6000+0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        plt.xticks(rotation=30)
        ax.legend(fontsize=18, ncol = 1, loc = 'best')
        ax.tick_params(axis='x', labelsize = 50)
        ax.tick_params(axis='y', labelsize = 50) 
        plt.rc('font', size=60)
        ax.grid(True)
        # ax.set_xlabel('Date', fontsize = font) 
        ax.set_ylabel('$\kappa$', fontsize = 80) 
        fig.set_size_inches(w=25, h=12.5)
        plt.savefig(save_results_to + 'Kappa.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Kappa.png', dpi=300)  
    
    #%%
        
        fig, ax = plt.subplots() 
    #    ax.plot(t_f, Kappa, 'k-', lw=5)  
        ax.plot(date_total, np.transpose(np.asarray(Kappa1_records)[:,:,0]) , 'k-', lw=2)  
        # ax.set_xlim(0-0.5,180)
        ax.set_ylim(-0.021, 1.021)
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
        ax.set_ylabel('$\kappa_1$', fontsize = 80) 
        fig.set_size_inches(w=25, h=12.5)
        plt.savefig(save_results_to + 'Kappa1_rec.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Kappa1_rec.png', dpi=300)  
    
    
        fig, ax = plt.subplots() 
    #    ax.plot(t_f, Kappa, 'k-', lw=5)  
        ax.plot(date_total, np.transpose(np.asarray(Kappa2_records)[:,:,0]) , 'k-', lw=2)  
        # ax.set_xlim(0-0.5,180)
        ax.set_ylim(-0.021, 1.021)
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
        ax.set_ylabel('$\kappa_2$', fontsize = 80) 
        fig.set_size_inches(w=25, h=12.5)
        plt.savefig(save_results_to + 'Kappa2_rec.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Kappa2_rec.png', dpi=300)  
    
    
        fig, ax = plt.subplots() 
    #    ax.plot(t_f, Kappa, 'k-', lw=5)  
        ax.plot(date_total, np.transpose(np.asarray(Kappa3_records)[:,:,0]) , 'k-', lw=2)  
        # ax.set_xlim(0-0.5,180)
        ax.set_ylim(-0.021, 1.021)
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
        ax.set_ylabel('$\kappa_3$', fontsize = 80) 
        fig.set_size_inches(w=25, h=12.5)
        plt.savefig(save_results_to + 'Kappa3_rec.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Kappa3_rec.png', dpi=300)  
    
    
        fig, ax = plt.subplots() 
    #    ax.plot(t_f, Kappa, 'k-', lw=5)  
        ax.plot(date_total, np.transpose(np.asarray(Kappa4_records)[:,:,0]) , 'k-', lw=2)  
        # ax.set_xlim(0-0.5,180)
        ax.set_ylim(-0.021, 1.021)
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
        ax.set_ylabel('$\kappa_4$', fontsize = 80) 
        fig.set_size_inches(w=25, h=12.5)
        plt.savefig(save_results_to + 'Kappa4_rec.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Kappa4_rec.png', dpi=300)   
    
    #%%
        fig, ax = plt.subplots() 
    #    ax.plot(t_f, Kappa, 'k-', lw=5)  
        ax.plot(date_total, Beta , 'k-', lw=2)  
        # ax.set_xlim(0-0.5,180)
        # ax.set_ylim(-0.021, 1.021)
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
        ax.set_ylabel(r'$\beta$', fontsize = 80) 
        fig.set_size_inches(w=25, h=12.5)
        plt.savefig(save_results_to + 'Beta.pdf', dpi=300) 
        plt.savefig(save_results_to + 'Beta.png', dpi=300)   
