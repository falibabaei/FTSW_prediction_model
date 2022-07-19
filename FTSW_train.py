


import pandas as pd
import numpy as np
import os
import random
import time
from pickle import dump



from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras import layers

from Some_usfull_classes import data_preprocessing, data_split, r_square,rmse
import LSTM_model
import hyperparameters_FTSW as params


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

  


   
#split the data into the input of the model and true value of the yield
X_train,y_train =data_split(df, params.seq_len,params.output_len, params.n_out)

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
X_train_minmax=data.values[:,:-params.n_out]
y_train=data.values[:len(y_train),-params.n_out] 
X_train=np.reshape(X_train_minmax,(X_train.shape[0],X_train.shape[1],X_train.shape[2]))

X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)




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

history=model.fit(X_train,y_train,batch_size=params.Batch_size,
                         epochs=params.Epoch, 
                         validation_data=(x_val,y_val),
                         verbose=2, 
                         callbacks=[checkpoint,tensorboard,Early]) 



