# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:07:16 2022

@author: Fahimeh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:49:50 2022

@author: Fahimeh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:03:27 2022

@author: Fahimeh
"""
#{'target': -0.21576160192489624, 'params': {'Batch_size': 34.138582500235756, 'Droup_out_size': 0.3240207630881275, 'decay': 0.009208682153565323, 'lr': 0.00664778636072961}}

#1. Variance inflation factor (VIF)

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
from pickle import dump


#40.86    |  0.09313  |  0.003456 |  0.003968 |  158.6 
#'Batch_size': 32.79142881704165, 'Droup_out_size': 0.175213393146741, 'decay': 0.0030113609127845665, 'lr': 0.0025856063178843302, 'node': 190.1111749315244}}


#ET0 as input
#{'target': -0.16582414507865906, 'params': {'Batch_size': 63.58844966195333, 'Droup_out_size': 0.09898179582791122, 'decay': 0.009472443141502201, 'lr': 0.008930129756216258, 'node': 38.20210945377355}}

seq_len=7 #(number of days for predicting ET)
Batch_size=40#50#40 #128

Epoch=250

lr= 0.003968#0.00025856#0.0003968 #2e-3
decay= 0.003456#0.0003011 #0.0003456#1e-3
hidden_units=158#190#158#256# #number of hidden neurons in LSTM layer 32,64,128,256,
dropout_size=0.09313# 0.17521#0.09313
n_out=1 
NAME = f"swc1_diario_biodagro-{int(time.time())}.h5"

dir_='D:/SM_estimation_paper'

df= pd.read_csv(os.path.join(dir_,'00203CEE_station_data_plot1_revers.csv'), encoding= 'unicode_escape', index_col='time')

df.drop(['dev_point (avg)',
       'dev_poin(Minuto)', 'SR (avg)', 'WS (avg)',  'LW (time)','VS (max)', 
       'VS (max).1','SM (20cm)', 'SM (40cm)',
       'SM (60cm)', 'SM (80cm)','summand' ,
        'VPD (avg)',  'tem(avg)','tem(max)', 'tem (Minuto)','RH (avg)','VPD (Minuto)',
       'RH (max)' ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)',], axis=1,inplace=True)


df.dropna(inplace=True)



  

#df= df.append(df_ilha2)
#df['SM (20cm)_target']=df['SM (20cm)']
#df['SM (20cm)']=df['SM (20cm)'].shift(1)
#df.dropna(inplace=True)
#df=df[['Per (sum)', 'tem(avg)', 'RH (avg)', 'SR (avg)', 'VPD (Minuto)',
 #      'SM (20cm)']]
def data_preprocessing(df):
    #normilized the data
    values=df.values
    scaler=StandardScaler()#MinMaxScaler()###
    
    values_normal=scaler.fit_transform(values)
   
  
    df=pd.DataFrame(values_normal, columns= df.columns, index=df.index)
    return df,scaler








def data_split(df):    
    sequential_data=[]
    prev_day = deque(maxlen=seq_len)
    y=deque(maxlen=seq_len)
    
    for i in df.values:
        prev_day.append([n for n in i[:-n_out]])
        y.append(i[-n_out:])
        
        if len(prev_day)==seq_len:
            sequential_data.append([np.array(prev_day),np.array(y)])
            
 
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


from tensorflow.keras import backend as K

def hard_swish(x):
    return x * (K.relu(x + 3., max_value = 6.) / 6.)

   
#split the data into the input of the model and true value of the yield
X,Y=data_split(df)

#X=np.concatenate((X,X2), axis=0)
#Y=np.concatenate((Y,Y2), axis=0)
indices = np.arange(len(X))
X_train, x_test, y_train, y_test,ind1,ind2 = train_test_split(X, Y,indices, test_size=0.2, random_state=42)
#train_size = int(len(X) * 0.8)


#X_train, x_test = X[0:train_size], X[train_size:len(X)]

#x_test = X[-47:len(X)]
#y_train, y_test = Y[0:train_size], Y[train_size:len(X)]
#y_test = Y[-47:len(X)]


X_train, y_train = shuffle(X_train, y_train )

# reshape the X_train and Y_train and concatenate them in order to have the appropriate shape for the data preprocessing function
#X_train_reshape=np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
#y_train=y_train.reshape(-1, 1)
#m=np.zeros((X_train.shape[0]*X_train.shape[1]-len(y_train),7))
#Y_train_reshape=np.concatenate((y_train,m),axis=0)
                                
#y_train_reshape=Y_train_reshape.reshape(-1, 1)
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

#X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)







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


#swc1_diario_biodagro-1648574345.h5
#R_square=0.9179239869117737
#2.196863733234349
#swc1_diario_biodagro-1648574940.h5
#R_square=0.9546141028404236
#1.633636812643348
#swc1_diario_biodagro-1648581502.h5
#R_square=0.8518097400665283
#2.9519218636493547
#model=tf.keras.models.load_model('swc1_diario_biodagro-1652107495.h5',  compile=False)
# testing the trained model:
#reshape and concatenate  the x_test and y_train  to do data preprocessing on the test set


#------------------------
#SWC40
#inputs:'tem(avg)', 'SR (avg)', 'RH (avg)', 'Precipitation', 'SM (20cm)',
#output:  'SM (40cm)'
#model:swc1_diario_biodagro-1648641160.h5

#-------------------summand-------------------
#swc1_diario_biodagro-1649667230.h5

#x_test_reshape=np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],x_test.shape[2]))



#m=np.zeros((x_test.shape[0]*x_test.shape[1]-len(y_test),1))
#y_test_reshape=np.concatenate((y_test,m),axis=0).reshape(-1, 1)
data1=np.concatenate((x_test,y_test),axis=2)

data2=np.reshape(data1,(data1.shape[0]*data1.shape[1],data1.shape[2]))
#convert the test set into a number between 0 and 1 using the Min_Max scaler from the training set
data1_re=scaler.transform(data2)
data1_re=np.reshape(data1_re,(data1.shape[0],data1.shape[1],data1.shape[2]))
x_test1=data1_re[:,:,:-n_out]
y_test1=data1_re[:,:,-n_out] 
#reshape the test set into the appropriate shape for prediction


#prediction using trained model
yhat = model.predict(x_test1)


#reshape and concatenate  the x_test and prediction  to denormalize the prediction

data3=np.concatenate((x_test,yhat),axis=2)
data3=np.reshape(data3,(data1.shape[0]*data1.shape[1],data1.shape[2]))
#denormalize the prediction and true value
data3=scaler.inverse_transform(data3)
data3=np.reshape(data3,(data1.shape[0],data1.shape[1],data1.shape[2]))
inv_yhat=data3[:,:,-n_out]
    
#inv_yhat=np.expand_dims(inv_yhat, axis=2)
y_test=np.reshape(y_test,(y_test.shape[0],y_test.shape[1]*y_test.shape[2]))

#r2_squer and mean squared error between the true values and the prediction values by model
r=R_squared(y_test,inv_yhat)
print(f'R_square={r}')       



print(np.sqrt(mean_squared_error(y_test,inv_yhat)))



fig, ax = plt.subplots(figsize=(20, 10))

# Add x-axis and y-axis.
ax.plot(y_test[3],
           color='purple', label='ilha1')
ax.plot(
         inv_yhat[3],
           label='ilha1')



fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( y_test,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(y_test, y_test, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Parcela 46-FTSW ")  # Add a title to the axes.
#ax.legend();  # Add a legend.
#plt.savefig(f'D:/SM_estimation_paper/sm_plots/ftsw_ilha1_VPDmin_RHmin_pre_ETo_Standars.jpg' ,dpi=500)




'''





df_ilha2=pd.read_csv(os.path.join(dir_,'summand_ilha2_daily.csv'), encoding= 'unicode_escape', index_col='time')




df_ilha2.drop([ 
        'LW (time)' ,'dev_point (avg)',
      'VS (max)',
       'VS (max).1','WS (avg)','dev_poin(Minuto)','tem(max)','RH (max)',  
     'RH (Minuto)',  'tem(avg)', 'SR (avg)', 'VPD (avg)', 'RH (avg)','FTSW1','summand','FTSW1'
      ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)','summand'], axis=1,inplace=True)

df_ilha2.dropna( inplace=True)
X,Y=data_split(df_ilha2)

#_, X, _, Y= train_test_split(X, Y, test_size=0.95, random_state=42)

#X=np.concatenate(( x_test,X), axis=0)
#Y=np.concatenate((y_test,Y), axis=0)

#X,Y=shuffle(X,Y)
x_test_reshape=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))



m=np.zeros((X.shape[0]*X.shape[1]-len(Y),1))
y_test_reshape=np.concatenate((Y,m),axis=0).reshape(-1, 1)
data1=np.concatenate((x_test_reshape,y_test_reshape),axis=1)

data1=pd.DataFrame(data1)
#convert the test set into a number between 0 and 1 using the Min_Max scaler from the training set
data1_re=scaler.transform(data1)

x_test1=data1_re[:,:-n_out]
y_test1=data1_re[:len(Y),-n_out] 
#reshape the test set into the appropriate shape for prediction
x_test1=np.reshape(x_test1,(X.shape[0],X.shape[1],X.shape[2]))

#prediction using trained model
yhat = model.predict(x_test1)




yhat_reshape=np.concatenate((yhat,m),axis=0)
data2=np.concatenate((x_test_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value
data1=scaler.inverse_transform(data1_re)
data2=scaler.inverse_transform(data2)
inv_y=data1[:len(Y),-n_out]
inv_yhat=data2[:len(Y),-n_out]
    



#r2_squer and mean squared error between the true values and the prediction values by model
r=r_square(inv_y,inv_yhat)
print(f'R_square={r}')       



print(np.sqrt(mean_squared_error(inv_y,inv_yhat)))



fig, ax = plt.subplots(figsize=(20, 10))

# Add x-axis and y-axis.
ax.plot(inv_y,
           color='purple', label='ilha1')
ax.plot(
         inv_yhat,
           label='ilha1')

plt.show()


fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( inv_y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(inv_y, inv_y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Parcela 46-FTSW ")







# Permutation Feature Importance


def feature_importance(x_val,y_val,model_path):
    results=[]
    model=tf.keras.models.load_model(model_path,  compile=False)
    yhat= model.predict(x_val, verbose=0)
    baseline_mae= np.mean(np.abs( yhat-y_val ))
    results.append({'feature':'BASELINE','mae':baseline_mae})
    for k in tqdm(range(x_val.shape[2])):#x_val.shape[2]=no.features
        sav_col=x_val[:,:,k].copy()
        np.random.shuffle(x_val[:,:,k])
        #x_val[:,:,k]=np.zeros((x_val[:,:,k].shape))
        yhat= model.predict(x_val, verbose=0)
        mae = np.mean(np.abs( yhat-y_val ))
        results.append({'feature':df.columns[k],'mae':mae})
        x_val[:,:,k] = sav_col
    return results,baseline_mae
def plot_feature_importance(results, baseline_mae,COLS):
            df1 = pd.DataFrame(results)
            df1 = df1.sort_values('mae')
            plt.figure(figsize=(20,10))
            plt.barh(np.arange(len(COLS)),df1.mae)
            plt.yticks(np.arange(len(COLS)),df1.feature.values)
            plt.title('LSTM Feature Importance',size=16)
            plt.ylim((-1,len(COLS)-1))
            plt.plot([baseline_mae,baseline_mae],[-1,len(COLS)-1], '--', color='orange',
                    label=f'Baseline Model \nMAE={baseline_mae:.3f}')
            plt.xlabel(f'MAE',size=14)
            plt.ylabel('Feature',size=14)
            plt.legend()
            plt.show() 
           # df.to_csv(df1,'D:/SM_estimation_paper/fearure_importance_lstm.csv')

results,baseline_mae= feature_importance(x_val,y_val,'swc1_diario_biodagro-1648649394.h5')   
plot_feature_importance(results,baseline_mae,COLS=df.columns)

'''