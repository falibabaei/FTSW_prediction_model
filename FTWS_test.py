# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:56:07 2022

@author: Fahimeh
"""
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from pickle import load

from Some_usfull_classes import data_preprocessing, data_split,r_square

import hyperparameters_FTSW as params
register_matplotlib_converters()


#open test set
dir_='D:/SM_estimation_paper/2003_mean.csv'
df=pd.read_csv(dir_, encoding= 'unicode_escape',index_col=['time'],parse_dates=['time'],#date column using the parameter parse_dates
                                      )
df.drop([ 
        'Gelo (h)',
             'H.R. >90 (h)', 
              'H.R. 80-90 (h)',
              'H.R. <40 (h)','Precip max (mm)', 'Humect','Temp -10cm ',
             'ET0model','eto', 'Temp max ', 'Temp min ', 'H.R. max ', 'H.R. min'
      ], axis=1,inplace=True)
df.dropna(inplace=True)

#load  the scaler from training set
scaler = load(open('scaler.pkl', 'rb'))


 X,Y=data_split(df, params.seq_len, params.output_len, params.n_out)


X_train_reshape=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
#y_train=y_train.reshape(-1, 1)
m=np.zeros((X.shape[0]*X.shape[1]-len(Y),1))
Y_train_reshape=np.concatenate((Y,m),axis=0)
                                
y_train_reshape=Y_train_reshape.reshape(-1, 1)
data=np.concatenate((X_train_reshape,y_train_reshape),axis=1)
data=pd.DataFrame(data)

#normalize the test set using scaler from training set

data=scaler.transform(data)

#Again reshape the normalized data into the appropriate shape for input and output of the model
X_train_minmax=data[:,:-n_out]
y=data[:len(Y),-n_out] 
x=np.reshape(X_train_minmax,(X.shape[0],X.shape[1],X.shape[2]))
    
# load the trained model
model=tf.keras.models.load_model('swc1_diario_biodagro-1657190424.h5',  compile=False)

yhat = model.predict(x)

#reshape and concatenate  the x_test and prediction  to denormalize the prediction
yhat_reshape=np.concatenate((yhat,m),axis=0)
data2=np.concatenate((X_train_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value
data2=scaler.inverse_transform(data2)
inv_yhat=data2[:len(Y),-n_out]
    
#Calculate the metrices

print('RMSE=',rmse(Y,inv_yhat))

print('R-square=',r_square(Y,inv_yhat))

#plot the true valye vs. predicted value
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( Y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(Y, Y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Alenquer, Lisboa (ETo from ETo calculator) ")  # Add a title to the axes.
#ax.legend();  # Add a legend.
#plt.savefig(f'D:/SM_estimation_paper/sm_plots/2003_2004_lisboa_rh-per_etocal.jpg' ,dpi=500)




