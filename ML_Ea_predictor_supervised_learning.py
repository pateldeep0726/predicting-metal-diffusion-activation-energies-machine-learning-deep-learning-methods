  #import all the required libraries##
import os
import sys
from functools import partial

import ase
import numpy as np
import pandas as pd
from ase.visualize import view
import xlrd
from ase.io import read, write
#from ase.lattice.surface import add_adsorbate
from matplotlib import pyplot as plt
from openTSNE import TSNE
from dscribe.descriptors.soap import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from pymatgen.core.surface import generate_all_slabs, get_symmetrically_distinct_miller_indices, get_symmetrically_equivalent_miller_indices, SlabGenerator, Lattice, Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from itertools import combinations

import scipy.spatial as spatial, scipy.cluster.hierarchy as hc
import seaborn as sns
import logging
import matplotlib
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import linear_model
from sklearn.linear_model import Lasso


mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# dist = k2d(Kmat)

##Understanding Co agglomoration on clean W2TiC2##
#names_is = ['IS1',
#'IS2',
#'IS3',
#'IS4',
#'IS5',
#'IS6',
#'IS7',
#'IS8',
#'IS9',
#'IS10']

names_is = ['IS10',
'IS11',
'IS12',
'IS13',
'IS14',
'IS15',
'IS16',
'IS17',
'IS18',
'IS19']


# names_fs = ['FS1',
# 'FS2',
# 'FS3',
# 'FS4',
# 'FS5',
# 'FS6',
# 'FS7',
# 'FS8',
# 'FS9',
# 'FS10']

names_fs = ['FS10',
'FS11',
'FS12',
'FS13',
'FS14',
'FS15',
'FS16',
'FS17',
'FS18',
'FS19']

##IS##

ranges = np.zeros((len(names_is), 2), dtype=int)
conf_idx = np.zeros(len(names_is), dtype=int)

traj = []

for i,n in enumerate(names_is):
    print(n)
    frames = read(f'./{n}.xyz','::')
    for frame in frames:
        # wrap each frame in its box
        frame.wrap(eps=1E-10)

        # mask each frame so that descriptors are only centered on Co (#27) atoms
        mask = np.zeros(len(frame))
        mask[np.where(frame.numbers == 27)[0]] = 1
        frame.arrays['center_atoms_mask'] = mask

    ranges[i] = (len(traj), len(traj) + len(frames))
    conf_idx[i] = len(traj)
    traj = [*traj, *frames]

# energies of the simulation frames
#energy = np.array([1.011,0.443,0.583,0.407,0.578,0.828,0.172,0.294,0.257,0.687,1.208,0.441,0.389,0.694,1.114,0.603,0.600,0.170,1.532]) #Pd all#
#energy = np.array([1.011,0.443,0.583,0.407,0.578,0.828,0.172,0.294,0.257]) #Pd slabs#
#energy = np.array([0.687,1.208,0.441,0.389,0.694,1.114,0.603,0.600,0.170,1.532]) #Pd NP#
#energy = np.array([0.4578,0.3946,0.3630,0.3448,0.0322,0.9719,0.6021,0.8706,0.4685,0.2574,0.3143,0.2648,0.3197,0.0881,0.3696,0.2622,0.2620,0.1239]) #Ag all#
energy = np.array([0.4685,0.2574,0.3143,0.2648,0.3197,0.0881,0.3696,0.2622,0.2620,0.1239]) #Ag slabs#
#energy = np.array([1.251,0.650,0.563,0.707,0.951,0.482,0.321,0.967,0.200,0.171]) #Pt slabs#
#energy = np.array([1.1599,0.6744,0.0907,0.7377,0.6255,0.9806,0.3421,0.2203,0.2047,0.2448]) #Rh slabs#

print(energy)

# extrema for the energies
max_e = max(energy)
min_e = min(energy)

hypers = {
    "r_cut": 18,
    "n_max": 5,
    "l_max": 0,
    "sigma": 0.1,
    "species": [47],
    "sparse": False,
    "periodic": True
}
surface_sites_list = []
soap = SOAP(**hypers)
normalizer = StandardScaler()

center_sp_list = ['Ag']

#soaps = normalizer.fit_transform(soap.transform(traj).get_features(soap))
soaps_is = np.array(soap.create(traj),dtype=object)
#print(soaps)
split_soaps_is = np.split(soaps_is, len(traj))
mean_soaps_is = np.mean(split_soaps_is, axis=1)

conf_split_soaps_is = np.array([split_soaps_is[ci] for ci in conf_idx])
conf_mean_soaps_is = np.mean(conf_split_soaps_is, axis=1)

# saving soap vectors
#np.savez('./soap_vectors.npz',
#         mean_soaps=mean_soaps,
#         soaps=soaps,
#         conf_split_soaps=conf_split_soaps,
#         conf_mean_soaps=conf_mean_soaps)

#print(soaps_is.shape)
tdesc_is=[]
for i in range(len(traj)):
    print(i)
    d = soaps_is[i]
    for j in range(len(d)):
        tdesc_is.append(d[j])
tdesc_is=np.array(tdesc_is)
soapdesc_is = normalize(tdesc_is, axis=1) # normalize
np.shape(soapdesc_is)


##FS##

ranges = np.zeros((len(names_fs), 2), dtype=int)
conf_idx = np.zeros(len(names_fs), dtype=int)

traj = []
for i,n in enumerate(names_fs):
    print(n)
    frames = read(f'./{n}.xyz','::')
    for frame in frames:
        # wrap each frame in its box
        frame.wrap(eps=1E-10)

        # mask each frame so that descriptors are only centered on Co (#27) atoms
        mask = np.zeros(len(frame))
        mask[np.where(frame.numbers == 27)[0]] = 1
        frame.arrays['center_atoms_mask'] = mask

    ranges[i] = (len(traj), len(traj) + len(frames))
    conf_idx[i] = len(traj)
    traj = [*traj, *frames]

# hypers = {
#     "r_cut": 1,
#     "n_max": 6,
#     "l_max": 2,
#     "sigma": 0.1,
#     "species": [79],
#     "sparse": False,
#     "periodic": True
# }
# surface_sites_list = []
# soap = SOAP(**hypers)
# normalizer = StandardScaler()

# center_sp_list = ['Au']

#soaps = normalizer.fit_transform(soap.transform(traj).get_features(soap))
soaps_fs = np.array(soap.create(traj),dtype=object)
#print(soaps)
split_soaps_fs = np.split(soaps_fs, len(traj))
mean_soaps_fs = np.mean(split_soaps_fs, axis=1)

conf_split_soaps_fs = np.array([split_soaps_fs[ci] for ci in conf_idx])
conf_mean_soaps_fs = np.mean(conf_split_soaps_fs, axis=1)

# saving soap vectors
#np.savez('./soap_vectors.npz',
#         mean_soaps=mean_soaps,
#         soaps=soaps,
#         conf_split_soaps=conf_split_soaps,
#         conf_mean_soaps=conf_mean_soaps)

#print(soaps_is.shape)
tdesc_fs=[]
for i in range(len(traj)):
    d = soaps_fs[i]
    for j in range(len(d)):
        tdesc_fs.append(d[j])
tdesc_fs=np.array(tdesc_fs)
soapdesc_fs = normalize(tdesc_fs, axis=1) # normalize
np.shape(soapdesc_fs)

##### Random Forest Regression in Python #######
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(2)
    tf.random.set_seed(2)
    np.random.seed(2)

X = np.concatenate((conf_mean_soaps_is,conf_mean_soaps_fs),axis=1)
nsamples, nx, ny = X.shape
X = X.reshape((nsamples,nx*ny))
print(len(X))
y = np.array(energy)

#Split the data into training and testing set

kf = KFold(n_splits=10)
print(kf)
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

Input_Shape = [X.shape[1]]

RF_pred = [None]*len(y)
Train_RF_pred = [None]*len(y)*10

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    print(train_index)
    
    reset_random_seeds()
    RegModel = RandomForestRegressor(n_estimators=200,criterion='squared_error')
    RF=RegModel.fit(X_train,y_train)
    pred_values = RF.predict(X_test)
    train_pred_values = RF.predict(X_train)
    
    for i,v in enumerate(pred_values):
        RF_pred[(i + count*int(len(y)/10))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_RF_pred[(i + count*int(len(y)/10))] = v
    
 #   RF_DF = pd.DataFrame(RF.history)

 #   RF_DF.loc[:, ['loss','val_loss']].plot()
 #   RF_DF.savefig('./val_loss_random_forest.png')
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(RF_pred[(0+int(len(y)/10)*count):(int(len(y)/10) + int(len(y)/10)*count)],y_test)
    CV_scores.append(test_score)
    print(count)
    train_score = mean_absolute_error(Train_RF_pred[(0+int(len(y)/10)*count):(9*int(len(y)/10) + int(len(y)/10)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,RF_pred[(0+int(len(y)/10)*count):(int(len(y)/10) + int(len(y)/10)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_RF_pred[(0+int(len(y)/10)*count):(9*int(len(y)/10) + int(len(y)/10)*count)])
    Train_r2.append(train_r2_score)
    print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, RF.predict(X)))
# print('R2 Value:',np.mean(Train_r2))

#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs((y - RF.predict(X)))) * 100))
# print('Accuracy',100- ((np.mean(CV_scores))*100))

fig,ax = plt.subplots(1)
ax.scatter(y,RF.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual Ea (eV)')
ax.set_ylabel('Predicted Ea (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(RF)), fontsize=16)
plt.text(0.00, 0.80, 'R^2: {:.2f}'.format(metrics.r2_score(y, RF.predict(X))), fontsize=14)
plt.text(0.00, 0.70, 'MAE : {:.2f}'.format(np.mean(np.abs(y - RF.predict(X)))), fontsize=14)
plt.text(0.00, 0.60, 'MAX : {:.2f}'.format(np.max(np.abs(y - RF.predict(X)))), fontsize=14)
#plt.text(-0.5, -1.2, 'MAE for blind set (eV): {:.2f}'.format(np.mean(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
#plt.text(-0.5, -1.1, 'MAX for blind set (eV): {:.2f}'.format(np.max(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
plt.draw()
plt.savefig(f'./actual_vs_pred_RF_split')

#Printing some sample values of prediction
print(y,RF.predict(X))


####### Neural Network algorithm #######

X = np.concatenate((conf_mean_soaps_is,conf_mean_soaps_fs),axis=1)
nsamples, nx, ny = X.shape
X = X.reshape((nsamples,nx*ny))
X = np.asarray(X).astype('float32')
y = np.array(energy)


kf = KFold(n_splits=10)
reset_random_seeds()
Input_Shape = [X.shape[1]]

# standardize dataset

NN_pred = [None]*len(y)
Train_NN_pred = [None]*len(y)*10

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    print(train_index)
    
    reset_random_seeds()
    
    NN_model = keras.Sequential([
    layers.BatchNormalization(input_shape = Input_Shape),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(128, activation = 'sigmoid'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(64, activation = 'sigmoid'),
    layers.Dense(1)
    ])

    NN_model.compile(optimizer = 'adam', loss = 'mae')
    
    history = NN_model.fit(X_train,y_train, batch_size = 100, epochs = 600, validation_split = 0.2, verbose = 0)
    pred_values = NN_model.predict(X_test)
    train_pred_values = NN_model.predict(X_train)
    
    for i,v in enumerate(pred_values):
        NN_pred[(i + count*int(len(y)/10))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_NN_pred[(i + count*int(len(y)/10))] = v
    
    history_DF = pd.DataFrame(history.history)

    history_DF.loc[:, ['loss','val_loss']].plot()
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(NN_pred[(0+int(len(y)/10)*count):(int(len(y)/10) + int(len(y)/10)*count)],y_test)
    CV_scores.append(test_score)
    
    train_score = mean_absolute_error(Train_NN_pred[(0+int(len(y)/10)*count):(9*int(len(y)/10) + int(len(y)/10)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,NN_pred[(0+int(len(y)/10)*count):(int(len(y)/10) + int(len(y)/10)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_NN_pred[(0+int(len(y)/10)*count):(9*int(len(y)/10) + int(len(y)/10)*count)])
    Train_r2.append(train_r2_score)
    print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, NN_model.predict(X)))

#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs(y - NN_model.predict(X)))*100))


fig,ax = plt.subplots(1)
ax.scatter(y,NN_model.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(NN_model)), fontsize=16)
plt.text(0.00, 0.80, 'R^2: {:.2f}'.format(metrics.r2_score(y, NN_model.predict(X))), fontsize=14)
plt.text(0.00, 0.70, 'MAE : {:.2f}'.format(np.mean(np.abs(y - NN_model.predict(X)))), fontsize=14)
plt.text(0.00, 0.60, 'MAX : {:.2f}'.format(np.max(np.abs(y - NN_model.predict(X)))), fontsize=14)

#plt.text(-0.5, -1.2, 'MAE for blind set (eV): {:.2f}'.format(np.mean(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
#plt.text(-0.5, -1.1, 'MAX for blind set (eV): {:.2f}'.format(np.max(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
plt.draw()
plt.savefig(f'./actual_vs_pred_NN_split')

