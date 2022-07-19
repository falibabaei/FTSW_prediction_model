
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from pickle import load
import sys

yamnet_base = 'C:/Users/Asus/.spyder-py3/gitlab'
sys.path.append(yamnet_base)

import preprocessing as pro
import hyperparameters_ET0 as param
from Some_usfull_classes import data_preprocessing, data_split, r_square,rmse



dir_='D:/SM_estimation_paper/2003_mean.csv'
df=pd.read_csv(dir_, encoding= 'unicode_escape')
df.drop([ 'time',
         'Gelo (h)', 'H.R. >90 (h)', 'H.R. 80-90 (h)',
              'H.R. <40 (h)','Precip max (mm)', 'Humect', 'ET0model',
              'Temp -10cm ','Precipita (mm)', 'FTSW'
              
      ], axis=1,inplace=True)


X,Y=data_split(df)



X=np.array(X)
Y=np.array(Y)        
        
# reshape the data in order to normalized data with scaler from the training set

X_train_reshape=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
#y_train=y_train.reshape(-1, 1)
m=np.zeros((X.shape[0]*X.shape[1]-len(Y),1))
Y_train_reshape=np.concatenate((Y,m),axis=0)
 y_train_reshape=Y_train_reshape.reshape(-1, 1)
data=np.concatenate((X_train_reshape,y_train_reshape),axis=1)
data=pd.DataFrame(data)

#normalize the training set 
scaler=load(open('scaler_eto.pkl', 'rb'))
data=scaler.transform(data)

#Again reshape the normalized data into the appropriate shape for input and output of the model
X_train_minmax=data[:,:-params.n_out]
y=data[:len(Y),-params.n_out] 
x=np.reshape(X_train_minmax,(X.shape[0],X.shape[1],X.shape[2]))
    
#load the trained model
model=tf.keras.models.load_model('et0_biodagro-1652108950.h5',  compile=False)

#prediction of the model
yhat = model.predict(x)

#reshape and concatenate  the x_test and prediction  to denormalize the prediction
yhat_reshape=np.concatenate((yhat,m),axis=0)
data2=np.concatenate((X_train_reshape,yhat_reshape),axis=1)
#denormalize the prediction and true value


data2=scaler.inverse_transform(data2)
inv_yhat=data2[:len(Y),-params.n_out]
    

df1=pd.DataFrame()
df1['eto']=np.round(inv_yhat,1)

#save the ET predicted by the model as a csv file
df1.to_csv('D:/SM_estimation_paper/2003_ETO.csv')

#calculate the metrics

rmse=rmse(Y,inv_yhat)

r2=r_square(Y,inv_yhat)



#plot the true value vs. predited value
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter( Y,inv_yhat, label='ilha1')  # Plot some data on the axes.
ax.plot(Y, Y, label='ilha2', color='yellow', linestyle="--")  # Plot more data on the axes...

ax.set_xlabel('True value')  # Add an x-label to the axes.
ax.set_ylabel('Predicted value')  # Add a y-label to the axes.
ax.set_title(f"Parcela 46-FTSW ")  # Add a title to the axes.
#ax.legend();  # Add a legend.
#plt.savefig(f'D:/SM_estimation_paper/sm_plots/2003_2004_lisboa.jpg' ,dpi=500)





