# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:00:20 2022

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import os
from collections import deque
import random
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PowerTransformer
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils import shuffle

#40.86    |  0.09313  |  0.003456 |  0.003968 |  158.6 
#'Batch_size': 32.79142881704165, 'Droup_out_size': 0.175213393146741, 'decay': 0.0030113609127845665, 'lr': 0.0025856063178843302, 'node': 190.1111749315244}}


#ET0 as input
#{'target': -0.16582414507865906, 'params': {'Batch_size': 63.58844966195333, 'Droup_out_size': 0.09898179582791122, 'decay': 0.009472443141502201, 'lr': 0.008930129756216258, 'node': 38.20210945377355}}

seq_len=7 #(number of days for predicting ET)
Batch_size=50#50#40 #128

Epoch=500
lr= 0.0003968#0.00025856#0.0003968 #2e-3
decay= 0.0003456#0.0003011 #0.0003456#1e-3
hidden_units=158#190#158#256# #number of hidden neurons in LSTM layer 32,64,128,256,
dropout_size=0.09313# 0.17521#0.09313
n_out=1 
NAME = f"swc1_diario_biodagro-{int(time.time())}.h5"

dir_='D:/SM_estimation_paper'

df= pd.read_csv(os.path.join(dir_,'00203CEE_station_data_plot1_revers.csv'), encoding= 'unicode_escape', index_col='time')

df.dropna(inplace=True)
df.drop([ 'tem(avg)', 'tem(max)','dev_point (avg)',
       'dev_poin(Minuto)', 'SR (avg)', 'VPD (avg)', 'VPD (Minuto)', 'RH (max)', 
       'RH (Minuto)','LW (time)','VS (max)', 'VS (max).1','SM (20cm)', 'SM (40cm)',
       'SM (60cm)', 'SM (80cm)','summand' , 'WS (avg)'
       
       
      ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)',], axis=1,inplace=True)


df.dropna(inplace=True)

#df['SM (20cm)_target']=df['SM (20cm)']
df['FTSW_initial']=df['FTSW'].shift(1)
df.dropna(inplace=True)
df['FTSW_target']=df['FTSW']
df.dropna(inplace=True)
df.drop(['FTSW'], inplace=True, axis=1)
#df=df[['Per (sum)', 'tem(avg)', 'RH (avg)', 'SR (avg)', 'VPD (Minuto)',
 #      'SM (20cm)']]
def data_preprocessing(df):
    #normilized the data
    values=df.values
    scaler=StandardScaler()#MinMaxScaler()#
    
    values_normal=scaler.fit_transform(values)
   
  
    df=pd.DataFrame(values_normal, columns= df.columns, index=df.index)
    return df,scaler








def data_split(df):    
    sequential_data=[]
    prev_day = deque(maxlen=seq_len)
    initial_point=[]
    
    for ind,i in enumerate(df.values):
        prev_day.append([n for n in i[:-n_out-1]])
        initial_point.append(i[-n_out-1])
        
        if len(prev_day)==seq_len:
            sequential_data.append([np.array(prev_day),i[-n_out:]])
            
 
   # random.shuffle(sequential_data)
        
    X=[]             
    Y=[]
    X1=initial_point
    
    for seq , target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X), np.array(Y)  ,np.array(X1) 


def r_square(y_true, y_pred):
    y_true= tf.convert_to_tensor(y_true, np.float32)
  #  from keras import backend as K
    SS_res = tf.keras.backend.sum(tf.keras.backend.square( y_true-y_pred ))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square( y_true - tf.keras.backend.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + tf.keras.backend.epsilon()) )

def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


from tensorflow.keras import backend as K

def hard_swish(x):
    return x * (K.relu(x + 3., max_value = 6.) / 6.)



def creat_Lstm():
    input1=tf.keras.layers.Input(shape=(X_train.shape[1],X_train.shape[2]))
    input2=tf.keras.layers.Input(shape=(X1_train.shape[1]))
    x=tf.keras.layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True, activation='relu'))(input1)
    #,recurrent_dropout=dropout_size, 
                                     
                                         #,name='lstm1')))(input1)
   # x=layers.Activation(hard_swish)(x)
    x=layers.Dropout(dropout_size)(x)
    x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units,#,recurrent_dropout=dropout_size, 
                                      return_sequences=False,activation='relu'))(x)#)))(dropout)
   # x=layers.Activation(hard_swish)(x)
    x=layers.Dropout(dropout_size)(x)
    x =tf.keras.layers.concatenate([x, input2],axis=-1)
    x=layers.Dense(hidden_units, activation='relu')(x)
    x=layers.Dropout(dropout_size)(x)
    output =tf.keras.layers.Dense(n_out)(x)
    
    model =tf.keras.models.Model(inputs=[input1,input2], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=lr,decay=decay)
    model.compile(optimizer=opt, loss='mse', metrics=["RootMeanSquaredError", r_square])
    return model


#split the data into the input of the model and true value of the yield
X,Y,X1=data_split(df)
X1=X1[0:X.shape[0]]
X, Y,X1 = shuffle(X, Y,X1)


indices = np.arange(len(X))
X_train, x_test, y_train, y_test,X1_train,X1_test = train_test_split(X, Y,X1, test_size=0.1, random_state=42)

X1_train=np.expand_dims(X1_train,axis=-1)

# reshape the X_train and Y_train and concatenate them in order to have the appropriate shape for the data preprocessing function
X_train_reshape=np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
#y_train=y_train.reshape(-1, 1)
m=np.zeros((X_train.shape[0]*X_train.shape[1]-len(y_train),1))
Y_train_reshape=np.concatenate((y_train,m),axis=0)
                                
y_train_reshape=Y_train_reshape.reshape(-1, 1)

X1_train_reshape=np.concatenate((X1_train,m),axis=0)


data=np.concatenate((X_train_reshape,X1_train_reshape,y_train_reshape),axis=1)
data=pd.DataFrame(data)

#normalize the training set in order to use for training process
data, scaler=data_preprocessing(data)

#Again reshape the normalized data into the appropriate shape for input and output of the model
X_train_minmax=data.values[:,:-n_out-1]

y_train=data.values[:len(y_train),-n_out] 
x1_train=data.values[:len(y_train),-n_out-1] 
X_train=np.reshape(X_train_minmax,(X_train.shape[0],X_train.shape[1],X_train.shape[2]))

X_train, x_val, y_train, y_val , X1_train,X1_val= train_test_split(X_train, y_train,x1_train, test_size=0.2, random_state=42)

X1_train=np.expand_dims(X1_train,axis=-1)
X1_val=np.expand_dims(X1_val,axis=-1)



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
model.load_weights('swc1_diario_biodagro-1650374689.h5', by_name=True)
history=model.fit([X_train,X1_train],y_train,batch_size=Batch_size,
                         epochs=Epoch, 
                        # validation_data=([x_val,X1_val],y_val),
                        validation_split=0.2,
                         verbose=2,
                        callbacks=[Early,checkpoint,tensorboard]) 




#swc1_diario_biodagro-1648574345.h5
#R_square=0.9179239869117737
#2.196863733234349
#swc1_diario_biodagro-1648574940.h5
#R_square=0.9546141028404236
#1.633636812643348
#swc1_diario_biodagro-1648581502.h5
#R_square=0.8518097400665283
#2.9519218636493547
#model=tf.keras.models.load_model('swc1_diario_biodagro-1649941117.h5',  compile=False)
# testing the trained model:
#reshape and concatenate  the x_test and y_train  to do data preprocessing on the test set


#------------------------
#SWC40
#inputs:'tem(avg)', 'SR (avg)', 'RH (avg)', 'Precipitation', 'SM (20cm)',
#output:  'SM (40cm)'
#model:swc1_diario_biodagro-1648641160.h5

#-------------------summand-------------------
#swc1_diario_biodagro-1649667230.h5

x_test_reshape=np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],x_test.shape[2]))

X1_test=np.expand_dims(X1_test,axis=-1)


m=np.zeros((x_test.shape[0]*x_test.shape[1]-len(y_test),1))
y_test_reshape=np.concatenate((y_test,m),axis=0).reshape(-1, 1)


X1_test_reshape=np.concatenate((X1_test,m),axis=0).reshape(-1, 1)

data1=np.concatenate((x_test_reshape,X1_test_reshape,y_test_reshape),axis=1)

data1=pd.DataFrame(data1)
#convert the test set into a number between 0 and 1 using the Min_Max scaler from the training set
data1_re=scaler.transform(data1)

x_test1=data1_re[:,:-n_out-1]
y_test1=data1_re[:len(y_test),-n_out] 
x1_test1=data1_re[:len(y_test),-n_out-1] 
#reshape the test set into the appropriate shape for prediction
x_test1=np.reshape(x_test1,(x_test.shape[0],x_test.shape[1],x_test.shape[2]))


x1_test1=np.expand_dims(x1_test1,axis=-1)
#prediction using trained model


yhat = model.predict([x_test1, x1_test1])


#reshape and concatenate  the x_test and prediction  to denormalize the prediction
yhat_reshape=np.concatenate((yhat,m),axis=0)
data2=np.concatenate((x_test_reshape,X1_test_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value
data1=scaler.inverse_transform(data1_re)
data2=scaler.inverse_transform(data2)
inv_y=data1[:len(y_test),-n_out]
inv_yhat=data2[:len(y_test),-n_out]
    



#r2_squer and mean squared error between the true values and the prediction values by model
r=r_square(inv_y,inv_yhat)
print(f'R_square={r}')       



print(np.sqrt(mean_squared_error(inv_y,inv_yhat)))






fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( inv_y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(inv_y, inv_y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Parcela 46-FTSW ")  # Add a title to the axes.
#ax.legend();  # Add a legend.
#plt.savefig(f'D:/SM_estimation_paper/sm_plots/ftsw_ilha1.jpg' ,dpi=500)




df_ilha2=pd.read_csv(os.path.join(dir_,'summand_ilha2_daily.csv'), encoding= 'unicode_escape')




df_ilha2.drop([ 
       'tem(avg)', 'tem(max)','dev_point (avg)',
              'dev_poin(Minuto)', 'SR (avg)', 'VPD (avg)', 'VPD (Minuto)', 'RH (max)', 
              'RH (Minuto)','LW (time)','VS (max)', 'VS (max).1','summand' ,
              'WS (avg)','FTSW1'], axis=1,inplace=True)#
df_ilha2.dropna( inplace=True)




#df['SM (20cm)_target']=df['SM (20cm)']
df_ilha2['FTSW_initial']=df_ilha2['FTSW'].shift(1)
df_ilha2.dropna(inplace=True)
df_ilha2['FTSW_target']=df_ilha2['FTSW']
df_ilha2.dropna(inplace=True)
df_ilha2.drop(['FTSW'], inplace=True, axis=1)

df_ilha2.drop(['time'], inplace=True, axis=1)
X,Y,X1=data_split(df_ilha2)

X1=X1[0:X.shape[0]]

x_test_reshape=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))

X1=np.expand_dims(X1,axis=-1)

m=np.zeros((X.shape[0]*X.shape[1]-len(Y),1))
y_test_reshape=np.concatenate((Y,m),axis=0).reshape(-1, 1)
X1_test_reshape=np.concatenate((X1,m),axis=0).reshape(-1, 1)


data1=np.concatenate((x_test_reshape,X1_test_reshape,y_test_reshape),axis=1)

data1=pd.DataFrame(data1)
#convert the test set into a number between 0 and 1 using the Min_Max scaler from the training set
data1_re=scaler.transform(data1)



x_test1=data1_re[:,:-n_out-1]
y_test1=data1_re[:len(Y),-n_out] 
X1_test=data1_re[:len(Y),-n_out-1] 
#reshape the test set into the appropriate shape for prediction
x_test1=np.reshape(x_test1,(X.shape[0],X.shape[1],X.shape[2]))

#prediction using trained model
yhat = model.predict([x_test1,X1_test])




yhat_reshape=np.concatenate((yhat,m),axis=0)

data2=np.concatenate((x_test_reshape,X1_test_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value
data1=scaler.inverse_transform(data1_re)
data2=scaler.inverse_transform(data2)
inv_y=data1[:len(Y),-n_out]
inv_yhat=data2[:len(Y),-n_out]
    



#r2_squer and mean squared error between the true values and the prediction values by model
r=r_square(inv_y,inv_yhat)
print(f'R_square={r}')       



print(np.sqrt(mean_squared_error(inv_y,inv_yhat)))

y_mean=np.array([np.mean(inv_y) for i in range(len(Y))])
print(np.sqrt(mean_squared_error(inv_y,y_mean)))




fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( inv_y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(inv_y, inv_y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Parcela 46-FTSW ") 



'''


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