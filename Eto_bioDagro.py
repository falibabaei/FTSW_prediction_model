

import pandas as pd
import numpy as np
import os
from collections import deque
import random
import matplotlib.pyplot as plt
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

import hyperparametrs_ET0 as params
from Some_usfull_classes import data_preprocessing, data_split, r_square,rmse
import LSTM_model

NAME = f"swc1_diario_biodagro-{int(time.time())}.h5"

dir_='D:/SM_estimation_paper'
#open the dataset
df= pd.read_csv(os.path.join(dir_,'Etotrain.csv'), encoding= 'unicode_escape', index_col='time')

#drop the columns we do not need
df.drop([ 'dev_point (avg)',
       'dev_poin(Minuto)', 'SR (avg)', 'VPD (avg)', 'VPD (Minuto)','LW (time)','VS (max)', 'VS (max).1','SM (20cm)', 'SM (40cm)',
       'SM (60cm)', 'SM (80cm)','summand' , 'WS (avg)','Precipitation','FTSW', 'tem(max)', 'tem (Minuto)',
       
          'RH (max)',
       'RH (Minuto)' ], axis=1,inplace=True)
df.dropna(inplace=True)




#split the data into the input of the model and true value of the yield
X_train,y_train=data_split(df)

X_train, y_train = shuffle(X_train, y_train )

# reshape the X_train and Y_train and concatenate them in order to have the appropriate shape for the data preprocessing function
X_train_reshape=np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
#y_train=y_train.reshape(-1, 1)
m=np.zeros((X_train.shape[0]*X_train.shape[1]-len(y_train),1))
Y_train_reshape=np.concatenate((y_train,m),axis=0)
                                
y_train_reshape=Y_train_reshape.reshape(-1, 1)
data=np.concatenate((X_train_reshape,y_train_reshape),axis=1)
data=pd.DataFrame(data)

#normalize the training set in order to use for training process
data, scaler=data_preprocessing(data)

#Again reshape the normalized data into the appropriate shape for input and output of the model
X_train_minmax=data.values[:,:-n_out]
y_train=data.values[:len(y_train),-n_out] 
X_train=np.reshape(X_train_minmax,(X_train.shape[0],X_train.shape[1],X_train.shape[2]))


model=LSTM_model.creat_Lstm()
model.summary()

#define the checkpoin, early stoping and tensorboard
tensorboard=tf.keras.callbacks.TensorBoard(
    log_dir='D:/SM_estimation_paper/logs')
checkpoint = tf.keras.callbacks.ModelCheckpoint(NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
Early=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)
#model.load_weights('swc1_diario_biodagro-1652106970.h5')#('swc1_diario_biodagro-1649866152.h5')

#train the model
history=model.fit(X_train,y_train,batch_size=Batch_size,
                         epochs=Epoch, 
                         #validation_data=(x_val,y_val),
                         validation_split=0.2,
                         verbose=2, 
                         callbacks=[Early,checkpoint,tensorboard]) 

