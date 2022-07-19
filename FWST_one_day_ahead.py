


import pandas as pd

import numpy as np
import os
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import dump



from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras import layers

from Some_usfull_classes import data_preprocessing, data_split, r_square,rmse

#hyperparameters were selected by bayesian optimization
seq_len=7#(number of days for predicting ET)
Batch_size=40#50#40 #128

Epoch=210
lr= 0.003968#0.00025856#0.0003968 #2e-3
decay= 0.003456#0.0003011 #0.0003456#1e-3
hidden_units=158#190#158#256# #number of hidden neurons in LSTM layer 32,64,128,256,
dropout_size=0.09313# 0.17521#0.09313
n_out=1 
NAME = f"swc1_diario_biodagro-{int(time.time())}.h5"

dir_='D:/SM_estimation_paper'
#read dataset
df= pd.read_csv(os.path.join(dir_,'00203CEE_station_data_plot1_revers.csv'), encoding= 'unicode_escape', index_col='time')

df.drop(['dev_point (avg)',
       'dev_poin(Minuto)', 'SR (avg)', 'WS (avg)',  'LW (time)','VS (max)', 
       'VS (max).1','SM (20cm)', 'SM (40cm)',
       'SM (60cm)', 'SM (80cm)','summand' ,
        'VPD (avg)',  'tem(avg)', 'RH (avg)','tem(max)', 'tem (Minuto)',
       'RH (max)' ], axis=1,inplace=True)
df.dropna(inplace=True)

  
def creat_Lstm():
    input1=tf.keras.layers.Input(shape=(X_train.shape[1],X_train.shape[2]))
    x=tf.keras.layers.Bidirectional(layers.LSTM(hidden_units, 
                                                return_sequences=True, activation='relu', name='lstm1'))(input1)
    #,recurrent_dropout=dropout_size, 
                                     
                                         #,name='lstm1')))(input1)
   # x=layers.Activation(hard_swish)(x)
    x=layers.Dropout(dropout_size)(x)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units,#,recurrent_dropout=dropout_size, 
                                      return_sequences=False,activation='relu',name='lstm2'))(x)#)))(dropout)
   # x=layers.Activation(hard_swish)(x)
    x=layers.Dropout(dropout_size)(x)
   # danse=layers.Dense(256, activation='relu')(dropout)
   # dropout=layers.Dropout(dropout_size)(danse)
    output =tf.keras.layers.Dense(n_out, name='dense')(x)
    
    model =tf.keras.models.Model(inputs=input1, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=lr,decay=decay)
    model.compile(optimizer=opt, loss='mse', metrics=["RootMeanSquaredError", R_squared])
    return model


   
#split the data into the input of the model and true value of the yield
X,Y=data_split(df)



indices = np.arange(len(X))
X_train, x_test, y_train, y_test,ind1,ind2 = train_test_split(X, Y,indices, test_size=0.2, random_state=42)


X_train, y_train = shuffle(X_train, y_train )

# reshape the X_train and Y_train and concatenate them in order to have the appropriate shape for the data preprocessing function
X_train_reshape=np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
m=np.zeros((X_train.shape[0]*X_train.shape[1]-len(y_train),1))
Y_train_reshape=np.concatenate((y_train,m),axis=0)
                                
y_train_reshape=Y_train_reshape.reshape(-1, 1)
data=np.concatenate((X_train_reshape,y_train_reshape),axis=1)
data=pd.DataFrame(data)

#normalize the training set in order to use for training process
data, scaler=data_preprocessing(data)
dump(scaler, open('scaler.pkl', 'wb'))
#Again reshape the normalized data into the appropriate shape for input and output of the model
X_train_minmax=data.values[:,:-n_out]
y_train=data.values[:len(y_train),-n_out] 
X_train=np.reshape(X_train_minmax,(X_train.shape[0],X_train.shape[1],X_train.shape[2]))

#X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)




logdir =f'D:/SM_estimation_paper/logs\\swc1_diario_biodagro-{int(time.time())}'
if  not os.path.exists( logdir ) : 
    os.mkdir ( logdir ) 

model=creat_Lstm()
model.summary()

tensorboard=tf.keras.callbacks.TensorBoard(
    log_dir=logdir,  profile_batch = 100000000)
checkpoint = tf.keras.callbacks.ModelCheckpoint(NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
Early=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    verbose=1,
    patience=20,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)
#model.load_weights('swc1_diario_biodagro-1656680612.h5', by_name='True')#('swc1_diario_biodagro-1649866152.h5')

history=model.fit(X_train,y_train,batch_size=Batch_size,
                         epochs=Epoch, 
                         #validation_data=(x_val,y_val),
                         validation_split=0.2,
                         verbose=2, 
                         callbacks=[checkpoint,tensorboard]) 



