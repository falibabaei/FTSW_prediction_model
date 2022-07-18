

import pandas as pd

import numpy as np
import os
from collections import deque
import random
from tqdm import tqdm

from sklearn.utils import shuffle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PowerTransformer
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()




seq_len=7 #(number of days for predicting ET)
Batch_size=63#50#40 #128

Epoch=500
lr= 0.00089#0.00025856#0.0003968 #2e-3
decay= 0.00094#0.0003011 #0.0003456#1e-3
hidden_units=38 #number of hidden neurons in LSTM layer 32,64,128,256,
dropout_size=0.09313# 0.17521#0.09313
n_out=1 
NAME = f"swc1_diario_biodagro-{int(time.time())}.h5"

dir_='D:/SM_estimation_paper'

df= pd.read_csv(os.path.join(dir_,'Etotrain.csv'), encoding= 'unicode_escape', index_col='time')

df.dropna(inplace=True)
df.drop([ 'dev_point (avg)',
       'dev_poin(Minuto)', 'SR (avg)', 'VPD (avg)', 'VPD (Minuto)','LW (time)','VS (max)', 'VS (max).1','SM (20cm)', 'SM (40cm)',
       'SM (60cm)', 'SM (80cm)','summand' , 'WS (avg)','Precipitation','FTSW', 'tem(max)', 'tem (Minuto)',
       
          'RH (max)',
       'RH (Minuto)' ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)',], axis=1,inplace=True)


df.dropna(inplace=True)



def data_preprocessing(df):
    #normilized the data
    values=df.values
    scaler=StandardScaler()###
    
    values_normal=scaler.fit_transform(values)
   
  
    df=pd.DataFrame(values_normal, columns= df.columns, index=df.index)
    return df,scaler








def data_split(df):    
    sequential_data=[]
    prev_day = deque(maxlen=seq_len)
    
    for i in df.values:
        prev_day.append([n for n in i[:-n_out]])
        
        if len(prev_day)==seq_len:
            sequential_data.append([np.array(prev_day),i[-n_out:]])
            
 
  #  random.shuffle(sequential_data)
        
    X=[]             
    Y=[]
    
    for seq , target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X), np.array(Y)   


def r_square(y_true, y_pred):
    y_true= tf.convert_to_tensor(y_true, np.float32)
  #  from keras import backend as K
    SS_res = tf.keras.backend.sum(tf.keras.backend.square( y_true-y_pred ))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square( y_true - tf.keras.backend.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + tf.keras.backend.epsilon()) )
def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.math.truediv(residual, total))
  return r2
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))




def creat_Lstm():
    input1=tf.keras.layers.Input(shape=(X_train.shape[1],X_train.shape[2]))
    x=tf.keras.layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True, activation='relu'))(input1)
    #,recurrent_dropout=dropout_size, 
                                     
                                         #,name='lstm1')))(input1)
   # x=layers.Activation(hard_swish)(x)
    x=layers.Dropout(dropout_size)(x)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units,#,recurrent_dropout=dropout_size, 
                                      return_sequences=False,activation='relu'))(x)#)))(dropout)
   # x=layers.Activation(hard_swish)(x)
    x=layers.Dropout(dropout_size)(x)
   # danse=layers.Dense(256, activation='relu')(dropout)
   # dropout=layers.Dropout(dropout_size)(danse)
    output =tf.keras.layers.Dense(n_out)(x)
    
    model =tf.keras.models.Model(inputs=input1, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=lr,decay=decay)
    model.compile(optimizer=opt, loss='mse', metrics=["RootMeanSquaredError", R_squared])
    return model
   
#split the data into the input of the model and true value of the yield
X_train,y_train=data_split(df)


#X=np.concatenate((X,X2), axis=0)
#Y=np.concatenate((Y,Y2), axis=0)
#indices = np.arange(len(X))
#X_train, x_test, y_train, y_test,ind1,ind2 = train_test_split(X, Y,indices, test_size=0.2, random_state=42)


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

#X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)





model=creat_Lstm()
model.summary()
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
history=model.fit(X_train,y_train,batch_size=Batch_size,
                         epochs=Epoch, 
                         #validation_data=(x_val,y_val),
                         validation_split=0.2,
                         verbose=2, 
                         callbacks=[Early,checkpoint,tensorboard]) 



#------------------------
#SWC40
#inputs:'tem(avg)', 'SR (avg)', 'RH (avg)', 'Precipitation', 'SM (20cm)',
#output:  'SM (40cm)'
#model:swc1_diario_biodagro-1648641160.h5

#-------------------summand-------------------
#swc1_diario_biodagro-1649667230.h5

df_test= pd.read_csv(os.path.join(dir_,'ETOtestset.csv'), index_col='time')

df_test.dropna(inplace=True)
df_test.drop([ 'Precipitation', 'LW (time)', 'WS (avg)', 'VS (max)',
'VS (max).1','ETo calculator','tem(max)', 'tem (Minuto)', 'RH (max)', 'RH (Minuto)'
        ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)',], axis=1,inplace=True)


df.dropna(inplace=True)

x_test,y_test=data_split(df_test)
x_test_reshape=np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],x_test.shape[2]))



m=np.zeros((x_test.shape[0]*x_test.shape[1]-len(y_test),1))
y_test_reshape=np.concatenate((y_test,m),axis=0).reshape(-1, 1)
data1=np.concatenate((x_test_reshape,y_test_reshape),axis=1)

data1=pd.DataFrame(data1)
#convert the test set into a number between 0 and 1 using the Min_Max scaler from the training set
data1_re=scaler.transform(data1)

x_test1=data1_re[:,:-n_out]
y_test1=data1_re[:len(y_test),-n_out] 
#reshape the test set into the appropriate shape for prediction
x_test1=np.reshape(x_test1,(x_test.shape[0],x_test.shape[1],x_test.shape[2]))

#prediction using trained model
yhat = model.predict(x_test1)


#reshape and concatenate  the x_test and prediction  to denormalize the prediction
yhat_reshape=np.concatenate((yhat,m),axis=0)
data2=np.concatenate((x_test_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value
data1=scaler.inverse_transform(data1_re)
data2=scaler.inverse_transform(data2)
inv_y=data1[:len(y_test),-n_out]
inv_yhat=data2[:len(y_test),-n_out]
    



#r2_squer and mean squared error between the true values and the prediction values by model
r=R_squared(inv_y,inv_yhat)
print(f'R_square={r}')       


r=R_squared(inv_y,inv_yhat)


print(np.sqrt(mean_squared_error(inv_y,inv_yhat)))
print(np.sqrt(mean_squared_error(df_test['Daily ET0 [mm]'],df_test['ETo calculator'])))



fig, ax = plt.subplots(figsize=(20, 10))

# Add x-axis and y-axis.
ax.plot(inv_y,
           color='purple', label='ilha1')
ax.plot(
         inv_yhat,
           label='ilha1')
plt.savefig(f'D:/SM_estimation_paper/sm_plots/ftsw_ilha1_VPDmin_RHmin_pre_ETo_Standars.jpg' ,dpi=500)


fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(inv_y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(inv_y, inv_y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"ET0 predicted by LSTM")
plt.savefig(f'D:/SM_estimation_paper/sm_plots/Et0_model.jpg' ,dpi=500)
