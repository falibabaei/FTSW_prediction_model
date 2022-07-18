# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:52:51 2022

@author: Fahimeh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:56:07 2022

@author: Fahimeh
"""
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas.plotting import register_matplotlib_converters
import hyperparameters as param
import preprocessing as pro
register_matplotlib_converters()


from pickle import load
import sys
yamnet_base = 'C:/Users/Asus/.spyder-py3/gitlab'
sys.path.append(yamnet_base)


import hyperparameters as param
import preprocessing as pro
register_matplotlib_converters()



register_matplotlib_converters()

n_out=1
seq_len=7
dir_='D:/SM_estimation_paper/2003_mean.csv'
df=pd.read_csv(dir_, encoding= 'unicode_escape')
df.drop([ 'time',
         'Gelo (h)', 'H.R. >90 (h)', 'H.R. 80-90 (h)',
              'H.R. <40 (h)','Precip max (mm)', 'Humect', 'ET0model',
              'Temp -10cm ','Precipita (mm)', 'FTSW'
              
      ], axis=1,inplace=True)#,'RH (max)', 'RH (Minuto)', 'tem(max)', 'tem (Minuto)',
       #'Daily ET0 [mm]' ,'VPD (Minuto)',  'VPD (avg)', 'RH (avg)',], axis=1,inplace=True)




X1,Y1=pro.data_split_one_day_ahead(df)
X=[]
Y=[]
def min(a):
    return np.isnan(np.min(a))

for i , j in zip(X1,Y1):
    if min(j)!=True:
        X.append(i)
        Y.append(j)

X=np.array(X)
Y=np.array(Y)        
        

X_train_reshape=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
#y_train=y_train.reshape(-1, 1)
m=np.zeros((X.shape[0]*X.shape[1]-len(Y),1))
Y_train_reshape=np.concatenate((Y,m),axis=0)
                                
y_train_reshape=Y_train_reshape.reshape(-1, 1)
data=np.concatenate((X_train_reshape,y_train_reshape),axis=1)
data=pd.DataFrame(data)

#normalize the training set in order to use for training process
scaler=load(open('scaler_eto.pkl', 'rb'))
data=scaler.transform(data)
#data, scaler=data_preprocessing(data)

#Again reshape the normalized data into the appropriate shape for input and output of the model
X_train_minmax=data[:,:-n_out]
y=data[:len(Y),-n_out] 
x=np.reshape(X_train_minmax,(X.shape[0],X.shape[1],X.shape[2]))
    
# FWST: swc1_diario_biodagro-1651831879.h5'
model=tf.keras.models.load_model('swc1_diario_biodagro-1652108950.h5',  compile=False)

yhat = model.predict(x)

#reshape and concatenate  the x_test and prediction  to denormalize the prediction
yhat_reshape=np.concatenate((yhat,m),axis=0)
data2=np.concatenate((X_train_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value


data2=scaler.inverse_transform(data2)
inv_yhat=data2[:len(Y),-n_out]
    

df1=pd.DataFrame()
df1['eto']=np.round(inv_yhat,1)

df1.to_csv('D:/SM_estimation_paper/2003_ETO.csv')

print(np.sqrt(mean_squared_error(Y,inv_yhat)))
y_mean=np.array([np.mean(Y) for i in range(len(Y))])
mean_error=((mean_squared_error(Y,y_mean)))

mse=((mean_squared_error(Y,inv_yhat)))

r2=1-(mse/mean_error)
print(r2)



fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( Y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(Y, Y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Parcela 46-FTSW ")  # Add a title to the axes.
#ax.legend();  # Add a legend.
#plt.savefig(f'D:/SM_estimation_paper/sm_plots/2003_2004_lisboa.jpg' ,dpi=500)





