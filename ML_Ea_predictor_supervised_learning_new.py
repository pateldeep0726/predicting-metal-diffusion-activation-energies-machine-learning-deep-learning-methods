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
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from itertools import combinations
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import linear_model
from sklearn.linear_model import Lasso



def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   tf.random.set_seed(2)
   np.random.seed(2) 

##Inputs from LTR's bimetallic paper##
dataset = pd.read_excel("monometallic_all.xlsx",sheet_name="Rh")
dataset = dataset.dropna()
X = dataset.drop(columns = ['events','energy','ML predicted_RF','ABS Err','ML predicted_KR','ML predicted_BRR','ML predicted_GPR','ML predicted_LASSO','ML predicted_NN'])
Y = dataset['energy']

#Separate target variable and predictor variables##
targetvariable = 'energy'
predictors = ['a1-2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12']
X = dataset[predictors].values
y = dataset[targetvariable].values

#Split the data into training and testing set
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

###### Random Forest Regression in Python #######
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

#Train everything together#
#RegModel = RandomForestRegressor(n_estimators=100,criterion='friedman_mse',random_state=4277)
krr = KernelRidge(alpha=1.0)
KRR=krr.fit(X,y)
#predictions_all = np.array([tree.predict(X) for tree in RegModel.estimators_])


#Measuring Goodness of fit in Training data
from sklearn import metrics

#Plotting the feature importance for Top 10 most important columns

# feature_importances = pd.Series(RF.feature_importances_, index=predictors)
# plot1 = feature_importances.nlargest(20).plot(kind='barh')
# fig1 = plot1.get_figure()
# fig1.savefig("./feature_importances_Ag_RF.png")

#yplot = RF.predict(X[14:])
#xplot = y[14:]
DF = pd.DataFrame(KRR.predict(X))
DF.to_csv("predicted_KR.csv")

#GPR#
X = dataset[predictors].values
y = dataset[targetvariable].values
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)
GPR=gpr.fit(X,y)


DF = pd.DataFrame(GPR.predict(X))
DF.to_csv("predicted_GPR.csv")

#LASSO#
X = dataset[predictors].values
y = dataset[targetvariable].values
RegModel = Lasso()
RF=RegModel.fit(X,y)


DF = pd.DataFrame(RF.predict(X))
DF.to_csv("predicted_LASSO.csv")

####### Bayesian Ridge regression #######
from sklearn import linear_model
X = dataset[predictors].values
y = dataset[targetvariable].values
kernel = DotProduct() + WhiteKernel()
brr = linear_model.BayesianRidge()
BRR=brr.fit(X,y)
DF = pd.DataFrame(BRR.predict(X))
DF.to_csv("predicted_BRR.csv")

# fig,ax = plt.subplots(1)
# ax.scatter(xplot,yplot)
# ax.plot([0,2.0],[0,2.0])
# ax.set_xlabel('Actual (eV)')
# ax.set_ylabel('Predicted (eV)')
# ax.set_xlim([0,2.0])
# ax.set_ylim([0,2.0])
# ax1 = plt.gca()
# ax1.set_aspect('equal', adjustable='box')
# plt.title('{}'.format(str(RF)), fontsize=16)
# plt.text(0.5, 1.8, 'R^2: {:.2f}'.format(metrics.r2_score(xplot,yplot)), fontsize=14)
# plt.text(0.5, 1.7, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(xplot - yplot))), fontsize=14)
# plt.text(0.5, 1.6, 'MAX (eV): {:.2f}'.format(np.max(np.abs(xplot - yplot))), fontsize=14)
# #plt.text(-0.5, -3.4, 'MAE for blind set (eV): {:.2f}'.format(np.mean(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
# #plt.text(-0.5, -3.1, 'MAX for blind set (eV): {:.2f}'.format(np.max(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
# plt.draw()
# plt.savefig('./actual_vs_pred_Ag_Ea_RF')



#Printing some sample values of prediction
#TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
#TestingDataResults[targetvariable]=y_test
#TestingDataResults[('Predicted'+targetvariable)]=RF.predict(X_test)
#print(TestingDataResults)

# lax = [0,7]
# lax = np.array(lax).reshape((-1,1))
# prediction_test=RF.predict(lax)
# print(prediction_test)

####### Neural Network algorithm #######

##Train everything together##
X = dataset[predictors].values
y = dataset[targetvariable].values

reset_random_seeds()

Input_Shape = [X.shape[1]]

NN_model = keras.Sequential([
layers.BatchNormalization(input_shape = Input_Shape),
layers.Dense(512, activation = 'relu'),
layers.Dropout(0.3),
layers.BatchNormalization(),
layers.Dense(256, activation = 'relu'),
layers.Dropout(0.3),
layers.BatchNormalization(),
layers.Dense(128, activation = 'sigmoid'),
layers.Dropout(0.3),
layers.BatchNormalization(),
layers.Dense(64, activation = 'sigmoid'),
layers.Dense(1)
])

NN_model.compile(optimizer = 'adam', loss = 'mae')

history = NN_model.fit(X,y, batch_size = 100, epochs = 600, validation_split = 0.2, verbose = 0)

DF = pd.DataFrame(NN_model.predict(X))
DF.to_csv("predicted_NN.csv")

# yplot = NN_model.predict(X[14:])
# xplot = y[14:]
# fig,ax = plt.subplots(1)
# ax.scatter(xplot,yplot)
# ax.plot([0,2.0],[0,2.0])
# ax.set_xlabel('Actual (eV)')
# ax.set_ylabel('Predicted (eV)')
# ax.set_xlim([0,2.0])
# ax.set_ylim([0,2.0])
# ax1 = plt.gca()
# ax1.set_aspect('equal', adjustable='box')
# plt.title('{}'.format(str(RF)), fontsize=16)
# plt.text(0.5, 1.8, 'R^2: {:.2f}'.format(metrics.r2_score(xplot,yplot)), fontsize=14)
# plt.text(0.5, 1.7, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(xplot - yplot))), fontsize=14)
# plt.text(0.5, 1.6, 'MAX (eV): {:.2f}'.format(np.max(np.abs(xplot - yplot))), fontsize=14)
# #plt.text(-0.5, -3.4, 'MAE for blind set (eV): {:.2f}'.format(np.mean(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
# #plt.text(-0.5, -3.1, 'MAX for blind set (eV): {:.2f}'.format(np.max(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
# plt.draw()
# plt.savefig('./actual_vs_pred_Ag_Ea_NN')


# #Printing some sample values of prediction
# TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
# TestingDataResults[targetvariable]=y_test
# TestingDataResults[('Predicted'+targetvariable)]=NN_model.predict(X_test)
# print(TestingDataResults)