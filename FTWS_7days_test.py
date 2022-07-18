# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:48:56 2022

@author: Fahimeh
"""

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
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PowerTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from collections import deque

from sklearn.utils import shuffle
register_matplotlib_converters()

n_out=1
seq_len=7
dir_='D:/SM_estimation_paper/2003_mean.csv'
df=pd.read_csv(dir_, encoding= 'unicode_escape')
df.drop([ 'time',
        'Temp max ', 'Temp med ', 'Gelo (h)', 'H.R. min',
              'H.R. max ','H.R. >90 (h)', 'H.R. 80-90 (h)',
              'H.R. <40 (h)','Precip max (mm)', 'Humect','Temp -10cm ',
              'eto','Temp min ', 
      ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)',], axis=1,inplace=True)




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
 

X1,Y1=data_split(df)
X=[]
Y=[]
def min(a):
    return np.isnan(np.min(a))

for i , j in zip(X1,Y1):
    if min(j)!=True:
        print(j)
        X.append(i)
        Y.append(j)

X=np.array(X)
Y=np.array(Y)        
#X, Y= shuffle(X,Y )        
        

X_train_reshape=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
#y_train=y_train.reshape(-1, 1)
m=np.zeros((X.shape[0]*X.shape[1]-len(Y),1))
Y_train_reshape=np.concatenate((Y,m),axis=0)
                                
y_train_reshape=Y_train_reshape.reshape(-1, 1)
data=np.concatenate((X_train_reshape,y_train_reshape),axis=1)
data=pd.DataFrame(data)

#normalize the training set in order to use for training process
data=scaler.transform(data)
#data, scaler=data_preprocessing(data)

#Again reshape the normalized data into the appropriate shape for input and output of the model
X_train_minmax=data[:,:-n_out]
y=data[:len(Y),-n_out] 
x=np.reshape(X_train_minmax,(X.shape[0],X.shape[1],X.shape[2]))
    
# FWST: swc1_diario_biodagro-1651831879.h5'
model=tf.keras.models.load_model('swc1_diario_biodagro-1652176316.h5',  compile=False)

yhat = model.predict(x)

#reshape and concatenate  the x_test and prediction  to denormalize the prediction
yhat_reshape=np.concatenate((yhat,m),axis=0)
data2=np.concatenate((X_train_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value
data2=scaler.inverse_transform(data2)
inv_yhat=data2[:len(Y),-n_out]
    


def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.math.truediv(residual, total))
  return r2
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


#r2_squer and mean squared error between the true values and the prediction values by model
#r=R_squared(Y,inv_yhat)
#print(f'R_square={r}')       

y_mean=np.array([np.mean(Y) for i in range(len(Y))])
mean_error=((mean_squared_error(Y,y_mean)))

mse=((mean_squared_error(Y,inv_yhat)))

r2=1-(mse/mean_error)
print(r2)

print(np.sqrt(mean_squared_error(Y,inv_yhat)))

fig, ax = plt.subplots(figsize=(20, 10))

# Add x-axis and y-axis.
ax.plot(Y,
           color='purple', label='real value')
ax.plot(
         inv_yhat,
           label='Predictied')


fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( Y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(Y, Y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Alenquer, Lisboa (ETo from ETo calculator) ")  # Add a title to the axes.
#ax.legend();  # Add a legend.
#plt.savefig(f'D:/SM_estimation_paper/sm_plots/2003_2004_lisboa_rh-per_etocal.jpg' ,dpi=500)


'''

dir_='D:/SM_estimation_paper'
df_ilha2=pd.read_csv(os.path.join(dir_,'summand_ilha2_daily.csv'), encoding= 'unicode_escape', index_col='time')




df_ilha2.drop([ 
        'LW (time)' ,'dev_point (avg)',
      'VS (max)',
       'VS (max).1','WS (avg)','dev_poin(Minuto)','tem(max)','RH (max)',  
     'RH (Minuto)',  'tem(avg)', 'SR (avg)', 'VPD (avg)','FTSW1',
     'tem (Minuto)', 'VPD (Minuto)','summand','FTSW1'
      ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)','summand'], axis=1,inplace=True)

df_ilha2.dropna( inplace=True)
X,Y=data_split(df_ilha2)










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
'''