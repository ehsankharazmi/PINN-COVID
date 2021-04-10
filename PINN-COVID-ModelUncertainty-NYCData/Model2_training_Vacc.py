# -*- coding: utf-8 -*-
""" 


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
        self.PreS0 = U0[7]  
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
        self.Chi = tf.Variable(0.55,dtype=tf.float64,trainable=False)
        self.eps1 = tf.Variable(0.75,dtype=tf.float64,trainable=False) 
        self.eps2 = tf.Variable(0.0,dtype=tf.float64,trainable=False) 
        self.phiD = tf.Variable(1.0/15.0,dtype=tf.float64,trainable=False) 
        self.phiR = tf.Variable(1.0/7.5,dtype=tf.float64,trainable=False) 

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
        self.PreS0_u = tf.placeholder(tf.float64, shape=[None, self.PreS0.shape[1]]) 
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])   

        # physics informed neural networks
        self.S_pred, self.E_pred, self.P_pred, self.I_pred, self.J_pred, self.D_pred, self.H_pred, self.R_pred, self.I_sum_pred, self.D_sum_pred, self.H_sum_pred = self.net_u(self.t_u)  
        
        self.BetaI_pred = self.net_BetaI(self.t_u)          
        self.p_pred = self.net_p(self.t_u)  
        self.q_pred = self.net_q(self.t_u)  
        
        self.I_new_pred = self.I_sum_pred[1:,:]-self.I_sum_pred[0:-1,:]
        self.D_new_pred = self.D_sum_pred[1:,:]-self.D_sum_pred[0:-1,:]
        self.H_new_pred = self.H_sum_pred[1:,:]-self.H_sum_pred[0:-1,:]
        
        self.S0_pred = self.S_pred[0]
        self.E0_pred = self.E_pred[0] 
        self.I0_pred = self.I_pred[0]
        self.J0_pred = self.J_pred[0]
        self.D0_pred = self.D_pred[0]
        self.H0_pred = self.H_pred[0]  
        self.R0_pred = self.R_pred[0]  
        self.PreS0_pred = self.P_pred[0] 
        
        self.E_f, self.I_f, self.J_f, self.D_f, self.H_f, self.R_f, self.I_sum_f, self.H_sum_f, self.P_f = self.net_f(self.t_tf)
        
        # loss
        self.lossU0 = tf.reduce_mean(tf.square(self.E0_u - self.E0_pred)) + \
            tf.reduce_mean(tf.square(self.I0_u - self.I0_pred)) + \
            tf.reduce_mean(tf.square(self.J0_u - self.J0_pred)) + \
            tf.reduce_mean(tf.square(self.D0_u - self.D0_pred)) + \
            tf.reduce_mean(tf.square(self.H0_u - self.H0_pred)) + \
            tf.reduce_mean(tf.square(self.R0_u - self.R0_pred)) + \
            tf.reduce_mean(tf.square(self.PreS0_u - self.PreS0_pred))
            # tf.reduce_mean(tf.square(self.S0_u - self.S0_pred))  
        
        self.lossU = tf.reduce_mean(tf.square(self.I_new_u[:-1,:] - self.I_new_pred)) + \
            tf.reduce_mean(tf.square(self.D_new_u[:-1,:] - self.D_new_pred)) + \
            tf.reduce_mean(tf.square(self.H_new_u[:-1,:] - self.H_new_pred)) + \
            tf.reduce_mean(tf.square(self.I_sum_u - self.I_sum_pred)) + \
            tf.reduce_mean(tf.square(self.D_sum_u - self.D_sum_pred)) + \
            tf.reduce_mean(tf.square(self.H_sum_u - self.H_sum_pred)) 
            
        self.lossF = tf.reduce_mean(tf.square(self.E_f)) + tf.reduce_mean(tf.square(self.I_f)) + \
            tf.reduce_mean(tf.square(self.J_f)) + tf.reduce_mean(tf.square(self.D_f)) + \
            tf.reduce_mean(tf.square(self.H_f)) + tf.reduce_mean(tf.square(self.R_f)) + \
            tf.reduce_mean(tf.square(self.I_sum_f)) + tf.reduce_mean(tf.square(self.H_sum_f)) + \
            tf.reduce_mean(tf.square(self.P_f))

        self.loss = self.lossU0 + self.lossU + self.lossF 
        # self.loss = self.lossU + self.lossF 


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

    def net_u(self, t):
        SEIJDHR = self.neural_net(t, self.weights, self.biases)
        # SEIJDHR = SEIJDHR**2
        E = SEIJDHR[:,0:1] 
        I = SEIJDHR[:,1:2]
        J = SEIJDHR[:,2:3]
        D = SEIJDHR[:,3:4]
        H = SEIJDHR[:,4:5]
        R = SEIJDHR[:,5:6]   
        I_sum = SEIJDHR[:,6:7]
        H_sum = SEIJDHR[:,7:8]
        P = SEIJDHR[:,8:9]
        S = self.N-E-I-J-D-H-R-P
        D_sum = D 
        return S, E, P, I, J, D, H, R, I_sum, D_sum, H_sum
    
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
    
    def net_f(self, t):
        #load fixed parameters
        delta = self.delta 
        eps1 = self.eps1 
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        Chi = self.Chi
        gamma = self.gamma 
        gammaA = self.gammaA 
        phiD = self.phiD
        phiR = self.phiR 
        
        
        #load time-dependent parameters
        betaI = self.net_BetaI(t) 
        p = self.net_p(t)
        q = self.net_q(t)   
        
        #Obtain S,E,I,J,D,H,R from Neural network
        S, E, P, I, J, D, H, R, I_sum, D_sum, H_sum = self.net_u(t) 
        
        #Time derivatives 
        E_t = tf.gradients(E, t, unconnected_gradients='zero')[0]
        I_t = tf.gradients(I, t, unconnected_gradients='zero')[0]
        J_t = tf.gradients(J, t, unconnected_gradients='zero')[0]
        D_t = tf.gradients(D, t, unconnected_gradients='zero')[0]
        H_t = tf.gradients(H, t, unconnected_gradients='zero')[0]
        R_t = tf.gradients(R, t, unconnected_gradients='zero')[0]
        I_sum_t = tf.gradients(I_sum, t, unconnected_gradients='zero')[0]
        H_sum_t = tf.gradients(H_sum, t, unconnected_gradients='zero')[0]
        P_t = tf.gradients(P, t, unconnected_gradients='zero')[0]  

        #Residuals 
        VCn = tf.where(t<300*tf.ones(tf.shape(t), dtype = tf.float64), \
                       tf.zeros(tf.shape(t), dtype = tf.float64), \
                           tf.where(t<350*tf.ones(tf.shape(t), dtype = tf.float64), \
                           (t-300)*600, 30000*tf.ones(tf.shape(t), dtype = tf.float64)))#Vaccination
        VCn = VCn*self.sf
        f_E = E_t - (betaI*(Chi * P + I + eps1 * J)/self.N) * S + alpha1 * E
        f_I = I_t - (delta*alpha2) * P + gamma * I
        f_J = J_t - ((1-delta)*alpha2) * P + gammaA * J
        f_D = D_t - (q*phiD)*H
        f_H = H_t - (p*gamma)*I + (q*phiD) * H + ((1-q)*phiR) * H
        f_R = R_t - gammaA*J - ((1-p)*gamma)*I - ((1-q)*phiR)*H - VCn*S/self.N 
        f_I_sum = I_sum_t - (delta*alpha2)*E 
        f_H_sum = H_sum_t - (p*gamma)*I 
        f_P = P_t - alpha1* E + alpha2 * P

        return f_E, f_I, f_J, f_D, f_H, f_R, f_I_sum, f_H_sum, f_P  

    def callback(self, loss, lossU0, lossU, lossF):
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF]))
        print('Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e'  
              % (loss, lossU0, lossU, lossF))        
   
    def train(self, nIter):  
                
        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f, 
                   self.I_new_u: self.I_new_train, self.D_new_u: self.D_new_train, self.H_new_u: self.H_new_train, 
                   self.I_sum_u: self.I_sum_train, self.H_sum_u: self.H_sum_train, self.D_sum_u: self.D_sum_train,  
                   self.S0_u: self.S0, self.E0_u: self.E0, self.I0_u: self.I0, 
                   self.J0_u: self.J0, self.D0_u: self.D0, self.H0_u: self.H0, 
                   self.R0_u: self.R0, self.PreS0_u: self.PreS0}
        
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

        tf_dict = {self.t_u: t_star}
        
        S = self. sess.run(self.S_pred, tf_dict)
        E = self. sess.run(self.E_pred, tf_dict)
        P = self. sess.run(self.P_pred, tf_dict)
        I = self. sess.run(self.I_pred, tf_dict)
        J = self. sess.run(self.J_pred, tf_dict)
        D = self. sess.run(self.D_pred, tf_dict)
        H = self. sess.run(self.H_pred, tf_dict)
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
        return S, E, P, I, J, D, H, R, I_new, D_new, H_new, I_sum, D_sum, H_sum, BetaI, p, q 
        

############################################################    
if __name__=="__main__": 
  
    
    #Architecture of of the NN 
    layers=[1] + 10*[30] + [9] #The inout is t while the outputs rae E,I,J,D,H,R,I_sum,H_sum,D_sum 
    layers_beta=[1] + 5*[20] + [1] 
    layers_p=[1] + 5*[20] + [1]
    layers_q=[1] + 5*[20] + [1]

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
    PreS0 = I_sum_star[0:1,:]
    I0 = I_sum_star[0:1,:]
    J0 = PreS0 - I0
    D0 = D_sum_star[0:1,:]
    H0 = np.array([[0.0]])
    R0 = np.array([[0.0]])
    S0 = N - E0 - I0 - J0 - D0 - H0 - R0 - PreS0
    S0 = S0.reshape([1,1]) 
    U0 = [S0, E0, I0, J0, D0, H0, R0, PreS0]
    
    #Residual points 
    N_f = 3000
    t_f = lb + (ub-lb)*lhs(1, N_f) #uniformed grid for evaluating fractional derivative 
    

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

        relative_path_results = '/Model2/Train-Results-'+dt_string+'-'+casenumber+'/'
        save_results_to = current_directory + relative_path_results
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        relative_path = '/Model2/Train-model-'+dt_string+'-'+casenumber+'/' 
        save_models_to = current_directory + relative_path
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)
            break


    ####model 
    total_records = []
    total_records_LBFGS = []
    model = PhysicsInformedNN(t_train, I_new_train, D_new_train, H_new_train, 
                 I_sum_train, D_sum_train, H_sum_train, U0, t_f, lb, ub, N, 
                 layers, layers_beta, layers_p, layers_q, sf)


    ####Training 
    LBFGS=True
    # LBFGS=False
    model.train(10000) #Training with n iterations 

    ####save model 
    model.saver.save(model.sess, save_models_to+"model.ckpt")

    #Predicting   
    S, E, PreS, I, J, D, H, R, I_new, D_new, H_new, I_sum, D_sum, H_sum, BetaI, p, q = model.predict(t_star) 
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

    #save records
    np.savetxt(save_results_to +'iteration.txt', iteration.reshape((-1,1))) 
    np.savetxt(save_results_to +'loss_his.txt', loss_his.reshape((-1,1)))
    np.savetxt(save_results_to +'loss_his_u0.txt', loss_his_u0.reshape((-1,1))) 
    np.savetxt(save_results_to +'loss_his_u.txt', loss_his_u.reshape((-1,1))) 
    np.savetxt(save_results_to +'loss_his_f.txt', loss_his_f.reshape((-1,1)))   
    
    if LBFGS: 
        np.savetxt(save_results_to +'iteration_LBFGS.txt', iteration_LBFGS.reshape((-1,1))) 
        np.savetxt(save_results_to +'loss_his_LBFGS.txt', loss_his_LBFGS.reshape((-1,1)))
        np.savetxt(save_results_to +'loss_his_u0_LBFGS.txt', loss_his_u0_LBFGS.reshape((-1,1))) 
        np.savetxt(save_results_to +'loss_his_u_LBFGS.txt', loss_his_u_LBFGS.reshape((-1,1))) 
        np.savetxt(save_results_to +'loss_his_f_LBFGS.txt', loss_his_f_LBFGS.reshape((-1,1)))   

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
    if LBFGS: 
        plt.plot(iteration_LBFGS,loss_his_LBFGS, label='$loss-LBFGS$')
        plt.plot(iteration_LBFGS,loss_his_u0_LBFGS, label='$loss_{u0}-LBFGS$')
        plt.plot(iteration_LBFGS,loss_his_u_LBFGS, label='$loss_u-LBFGS$')
        plt.plot(iteration_LBFGS,loss_his_f_LBFGS, label='$loss_f-LBFGS$') 
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
    
#%%    
    #New cases 
    #Confirmed Cases
    font = 24
    fig, ax = plt.subplots()
    ax.plot(t_star, I_new_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
    # ax.plot(t_star, I_new_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
    ax.plot(t_star[:-1], I_new/sf, 'r-',lw=2,label='PINN') 
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

    #Hospitalized Cases
    font = 24
    fig, ax = plt.subplots()
    ax.plot(t_star, H_new_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
    # ax.plot(t_star, H_new_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
    ax.plot(t_star[:-1], H_new/sf, 'r-',lw=2,label='PINN') 
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize = 24)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.grid(True)
    ax.set_xlabel('Days', fontsize = font)
    ax.set_ylabel('New hospitalized ($H$)', fontsize = font) 
    fig.set_size_inches(w=13,h=6.5)
    if SAVE_FIG:
        plt.savefig(save_results_to +'new_hospitalized.png', dpi=300)  

    #Death Cases
    font = 24
    fig, ax = plt.subplots()
    ax.plot(t_star, D_new_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
    # ax.plot(t_star, D_new_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
    ax.plot(t_star[:-1], D_new/sf, 'r-',lw=2,label='PINN') 
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
    ax.plot(t_star, H_sum_star/sf, 'k--', marker = 'o',lw=2, markersize=5, label='Data-7davg')
    # ax.plot(t_star, H_sum_star/sf, 'b.', lw=2, markersize=5, label='Data-daily')
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
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize = 24)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.grid(True)
    ax.set_xlabel('Days', fontsize = font)
    ax.set_ylabel('$S$', fontsize = font) 
    fig.set_size_inches(w=13,h=6.5)
    if SAVE_FIG:
        plt.savefig(save_results_to +'Current_S.png', dpi=300) 
    
    font = 24
    fig, ax = plt.subplots()   
    ax.plot(t_star, E/sf, lw=2,label='E')   
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize = 24)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.grid(True)
    ax.set_xlabel('Days', fontsize = font)
    ax.set_ylabel('$E$', fontsize = font) 
    fig.set_size_inches(w=13,h=6.5)
    if SAVE_FIG:
        plt.savefig(save_results_to +'Current_E.png', dpi=300)
    
    font = 24
    fig, ax = plt.subplots()  
    ax.plot(t_star, J/sf, lw=2,label='J')   
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize = 24)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.grid(True)
    ax.set_xlabel('Days', fontsize = font)
    ax.set_ylabel('$J$', fontsize = font) 
    fig.set_size_inches(w=13,h=6.5)
    if SAVE_FIG:
        plt.savefig(save_results_to +'Current_J.png', dpi=300)
    
    font = 24
    fig, ax = plt.subplots()  
    ax.plot(t_star, R/sf, lw=2,label='R') 
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize = 24)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.grid(True)
    ax.set_xlabel('Days', fontsize = font)
    ax.set_ylabel('$R$', fontsize = font) 
    fig.set_size_inches(w=13,h=6.5)
    if SAVE_FIG:
        plt.savefig(save_results_to +'Current_R.png', dpi=300) 
        
    font = 24
    fig, ax = plt.subplots()  
    ax.plot(t_star, PreS/sf, lw=2,label='P')  
    # ax.set_xlim(0-0.5,180)
    # ax.set_ylim(0-0.5,6000+0.5)
    ax.legend(fontsize=22)
    ax.tick_params(axis='both', labelsize = 24)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    ax.grid(True)
    ax.set_xlabel('Days', fontsize = font)
    ax.set_ylabel('$P$', fontsize = font) 
    fig.set_size_inches(w=13,h=6.5)
    if SAVE_FIG:
        plt.savefig(save_results_to +'Current_P.png', dpi=300) 
