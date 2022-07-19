
import pandas as pd

import numpy as np
import os
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import dump


from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,r2_score

import tensorflow as tf


form  Some_usfull_classes import data_preprocessing, data_split, r_square,rmse 
#hyperparameters
seq_len=7 #(LSTM look back for prediction ET)
out_len=1 #(number of days that we predict the ET for them)
Batch_size=40#50#40 #128

Epoch=250

lr= 0.003968#0.00025856#0.0003968 #2e-3
decay= 0.003456#0.0003011 #0.0003456#1e-3
hidden_units=158#number of hidden neurons in LSTM layer 
dropout_size=0.09313
n_out=1 
NAME = f"swc1_diario_biodagro-{int(time.time())}.h5"

dir_='D:/SM_estimation_paper'

df= pd.read_csv(os.path.join(dir_,'00203CEE_station_data_plot1_revers.csv'), encoding= 'unicode_escape', index_col='time')

df.drop(['dev_point (avg)',
       'dev_poin(Minuto)', 'SR (avg)', 'WS (avg)',  'LW (time)','VS (max)', 
       'VS (max).1','SM (20cm)', 'SM (40cm)',
       'SM (60cm)', 'SM (80cm)','summand' ,
        'VPD (avg)',  'tem(avg)','tem(max)', 'tem (Minuto)','RH (avg)','VPD (Minuto)',
       'RH (max)' ], axis=1,inplace=True)

df.dropna(inplace=True)
   
#split the data into the input of the model and true value of the yield
X,Y=data_split(df)
indices = np.arange(len(X))
X_train, x_test, y_train, y_test,ind1,ind2 = train_test_split(X, Y,indices, test_size=0.2, random_state=42)

#shuffel training set 
X_train, y_train = shuffle(X_train, y_train )
data=np.concatenate((X_train,y_train),axis=2)
data1=np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))
data1=pd.DataFrame(data1)

#normalize the training set in order to use for training process
data1, scaler=data_preprocessing(data1)
dump(scaler, open('scaler.pkl', 'wb'))
#Again reshape the normalized data into the appropriate shape for input and output of the model

data=np.reshape(data1.values,(data.shape[0],data.shape[1],data.shape[2]))
X_train_minmax=data[:,:,:-n_out]
y_train=data[:,:,-n_out] 
X_train=X_train_minmax






def creat_Lstm():
    input1=tf.keras.layers.Input(shape=(X_train.shape[1],X_train.shape[2]))
    x=tf.keras.layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True, activation='relu'))(input1)
 
    x=layers.Dropout(dropout_size)(x)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units,#,recurrent_dropout=dropout_size, 
                                      return_sequences=True, activation='relu'))(x)
    x=layers.Dropout(dropout_size)(x)
    output =tf.keras.layers.Dense(n_out)(x)
    
    model =tf.keras.models.Model(inputs=input1, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=lr,decay=decay)
    model.compile(optimizer=opt, loss='mse', metrics=["RootMeanSquaredError", R_squared])
    return model




model=creat_Lstm()
model.summary()
tensorboard=tf.keras.callbacks.TensorBoard(
    log_dir='D:/SM_estimation_paper/logs')
checkpoint = tf.keras.callbacks.ModelCheckpoint(NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
Early=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=20,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)
#model.load_weights('swc1_diario_biodagro-1657532447.h5')#('swc1_diario_biodagro-1649866152.h5')

history=model.fit(X_train,y_train,batch_size=Batch_size,
                         epochs=Epoch, 
                         #validation_data=(x_val,y_val),
                         validation_split=0.2,
                         verbose=2, 
                         callbacks=[checkpoint,tensorboard]) 


