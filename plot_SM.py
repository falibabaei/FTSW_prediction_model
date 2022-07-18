# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 08:01:03 2022

@author: Fahimeh
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

base='D:\SM_estimation_paper/sm_plots/New folder/orginal_csv_file' 
dataframes_list = []
for i, csv in enumerate(os.listdir(base)):
    print(csv)
    path=os.path.join(base,csv)
    df=pd.read_csv(path)
    dataframes_list.append(df)
    
    
#plot sm20 for parcel26

    
# Sample data.

for nivel in ['20cm','40cm','60cm','80cm']:
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot( dataframes_list[0][nivel], label='ilha1')  # Plot some data on the axes.
    ax.plot(dataframes_list[1][nivel], label='ilha2')  # Plot more data on the axes...
    ax.plot( dataframes_list[2][nivel], label='ilha3')
    ax.plot( dataframes_list[3][nivel], label='ilha4')  # ... and some more.
    ax.set_xlabel('Time (Hourly)')  # Add an x-label to the axes.
    ax.set_ylabel('Soil moisture')  # Add a y-label to the axes.
    ax.set_title(f"Parcela 46-nivel {nivel} ")  # Add a title to the axes.
    ax.legend();  # Add a legend.
    plt.savefig(f'D:/SM_estimation_paper/sm_plots/parcela46-level{nivel}.jpg' ,dpi=500)
    
    
    
base='D:/A/Data/Fadagosa_2010_2020/Fadagosa_2010_2020.csv'
df=pd.read_csv(base,index_col='Time step' ) 

dir_='D:/Plots'
l=[r'$T_{Min}\  (^{\circ}C)$', r'$T_{Avg}\  (^{\circ}C)$', r'$T_{Max}\  (^{\circ}C)$',
   r'$HR_{Min}\  (\%)$', r'$HR_{Avg}\  (\%)$', r'$HR_{Max} \ (\%)$', r'$Prec \ (mm)$', 
   r'$SR \ (wm^{-2})$',
   r'$WS_{Avg}\ (ms^{-1})$']
for i,col in enumerate(df.columns):
    x  = np.arange(0,len(df))
    y=df[col] 
    plt.plot(x, y, label=l[i], c="red", lw=2)
#plt.legend()
    plt.xlabel('Time step')
    plt.ylabel(l[i])
 
    plt.savefig(os.path.join(dir_,col+'.pdf' ),dpi=500)
    plt.show()
