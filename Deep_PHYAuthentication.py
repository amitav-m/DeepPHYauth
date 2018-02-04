# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:58:14 2017

@author: Amitav Mukherjee
"""
import numpy as np
#import matplotlib.pyplot as plt
from scipy import linalg
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)

## Synthetic training set and corresponding labels generated and stored in 'Train'
tx_ant = 4 #Transmitter antennas at the devices
rx_ant = 4 #Receiver antennas at the authenticator
mH = np.zeros((rx_ant*tx_ant))
R_2 = np.zeros((rx_ant,rx_ant))
T_2 = np.zeros((tx_ant,tx_ant))
R_2f = np.zeros((rx_ant,rx_ant))
T_2f = np.zeros((tx_ant,tx_ant))
covH = np.eye(rx_ant*tx_ant) 
rho_t = 0.3; rho_r = 0.4; #for legitimate user in training set
rho_tf = 0.4; rho_rf = 0.5; #for spoofing user in training set
mu_K = 0
sigma_K = 0.1
mu_Kf = 0.2 #spoofing user
sigma_Kf = 0.03
train_setsize = 40000
test_setsize = 0
# Generate synthetic MIMO channels as training set
#H = sqrt(K/K+1)A + sqrt(1/(K+1))V where A is deterministic [On the Capacity Achieving Covariance Matrix for Rician MIMO Channels: An Asymptotic Approach]
# and V = R**0.5 x W x T**0.5, H is i.i.d. complex Gaussian and R,T are Tx/Rx correlation matrices. K is Rice factor and is random.
# Randomness of K factor: see arXiv:1306.3914v3

for m in range(1,rx_ant+1):
    for n in range(1,rx_ant+1):
        R_2[m-1,n-1] = rho_r**np.abs(m-n) # needs to be Nr x Nr
        R_2f[m-1,n-1] = rho_rf**np.abs(m-n) 
        #print(rho_t**np.abs(m-n))

for m in range(1,tx_ant+1):
    for n in range(1,tx_ant+1):
        T_2[m-1,n-1] = rho_t**np.abs(m-n)
        T_2f[m-1,n-1] = rho_tf**np.abs(m-n) 

R = np.asmatrix(linalg.sqrtm(R_2)) #legitimate user
T = np.asmatrix(linalg.sqrtm(T_2))
Rf = np.asmatrix(linalg.sqrtm(R_2f)) #spoofing user
Tf = np.asmatrix(linalg.sqrtm(T_2f))

## vec(ABC) = (C.T kron A)*vec(B), but make sure A,B,C are numpy matrices. In numpy, vec(A) -> A.flatten('F').T
h = np.asmatrix(np.random.multivariate_normal(mH,covH,train_setsize+test_setsize)) #i.i.d. channels
#W = np.reshape(h,(rx_ant,tx_ant)) # i.i.d. Gaussian matrix, but no need to actually reshape
## Now generate deterministic portion A as a Vandermonde matrix (ULA steering matrix). generated only once per user (legitimate/fake)
theta_aoa = np.random.uniform(0,1,tx_ant)  
vanin = np.exp(1j*theta_aoa)
A = np.matrix(np.vander(vanin, rx_ant, increasing=True))  
A_leg = A.T 

theta_aoa2 = np.random.uniform(0,1,tx_ant)  
vanin = np.exp(1j*theta_aoa2)
A = np.matrix(np.vander(vanin, rx_ant, increasing=True))  
A_fake = A.T 

#theta_aoa2 = np.random.uniform(0,1,tx_ant)  
#vanin = np.exp(1j*theta_aoa2)
#A = np.vander(vanin, rx_ant, increasing=True)  
#A_test = A.T # used during model evaluation for 'fake' user
  
K = np.random.normal(mu_K,sigma_K) 
Kf = np.random.normal(mu_Kf,sigma_Kf) 
##H = np.sqrt(K/K+1)*A + np.sqrt(1/K+1)*R*W*T
        
Train = np.empty([train_setsize+test_setsize,tx_ant*rx_ant+1], dtype='complex128') #half of train set from legitimate points and other half from synthetic 'fake' user. Last column is target label
for x in range(0,train_setsize+test_setsize):
    if x <= 0.5*train_setsize+test_setsize:
#        Train[x,0:16] =np.arange(16)
#         Train[x,0:16] =  np.transpose(np.sqrt(K/K+1)*(np.kron(T_2.T,R_2)*h[x,:].T))
        Train[x,0:tx_ant*rx_ant] = np.transpose(np.sqrt(K/K+1)*A_leg.flatten('F').T + np.sqrt(K/K+1)*(np.kron(T_2.T,R_2)*h[x,:].T))
        Train[x,-1] = 1 #Target label
    else:
        #Train[x,0:16] =np.arange(10,26)
        Train[x,0:tx_ant*rx_ant] = np.transpose(np.sqrt(Kf/Kf+1)*A_fake.flatten('F').T + np.sqrt(Kf/Kf+1)*(np.kron(T_2f.T,R_2f)*h[x,:].T))
        Train[x,-1] = 0 #Target label
        
np.random.shuffle(Train)   # shuffle target classes in training set          

Y = np.int(np.real(Train[:,-1]))  
Y = np.real(Train[:,-1]).astype(int)      

def create_baseline():
	# create model
    model = Sequential()    
    model.add(Dense(tx_ant*rx_ant, input_dim=tx_ant*rx_ant, kernel_initializer='normal', activation='relu'))    
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))    
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

# evaluate model via cross-fold validation on training set
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, Train[:,0:tx_ant*rx_ant], Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))        
#             