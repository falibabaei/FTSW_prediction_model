## Definition of the variables 

After the excess water has drained away and equilibrium has been reached, the field capacity (FC) denotes the amount of water remaining in the available soil volume.

The permanent wilting point (PWP) is reached when no more water is supplied to the soil, it gradually dries out, and the plant loses its freshness and withers.
In this work, FC and PWP were determined as the maximum and minimum soil moisture recorded by the probe, respectively.

The ratio between the soil water accessible at a given time (ASW) and the total transpirable soil water (TTSW) of a given crop in a given soil is called the fraction of transpirable soil water (FTSW). FTSW is used for irrigation scheduling in agriculture.

Evapotranspiration (ET) is the amount of water that evaporates from the Earth and plant surface. Solar radiation, wind speed, temperature, and relative humidity all influence daily ET, and this effect is highly non-linear.



## Long short term memory (LSTM)
 LSTM is a special type of RNN  designed for processing sequential data. 
 
## Feature Selection and data preprocessing

1-the missing  data were filled using the moving average method. 

2-The other preprocessing  involves the normalization.

3- The permutation Feature Importance technique was used to calculate the importance of the feature. The training set for this method is used to train a model, and the validation set is used to measure the increase in prediction error after a feature is permuted in the validation set, destroying the link between the feature and the true outcome. Since the model depends on the feature for prediction in this scenario, a feature is considered "important" if permuting its values results in an increase in model error. The feature with the highest importance (in this case, ET) was selected as the main independent variable. Then, the FEATURES were added to the group of independent variables one by one in order of importance, and the VIF value for this group of variables was calculated.
The variable is retained as a predictor if the VIF value is below a threshold; otherwise, it is removed from the group. After these processes, the historical data were selected as inputs to the FTSW prediction model, which included the month of the  month, day, RH (min) , VPD (min), precipitation, and daily ET. Since VPD (min) is not present in the Lisbon dataset, a model without this variable was created to evaluate the performance of the model on the test set. 



## Usage 
 
 The FWST model uses historical data, including relative humidity, ET, precipitation, month, and day of the month to predict FSTW for the next few days.

The Lisbon test set did not include ET, so we used the ET LSTM model to predict ET for that set.

## Installation
The model depends on the following Python packages:

numpy
tensorflow
sklearn
pandas
matplotlib

## Usage
The historical climate data are used to predict FTSW in the future.
## About the Model

The model code layout is as follows:

00203CEE_station_data_plot1_revers.csv: The file contains the dataset from colinas do douro (ilha1) (file is not included because the data  permission.) 

2003_mean.csv: test dataset from Lisbon (file is not included because the data  permission.) 

LSTM_model.py: A bidirectional two-layer model definition in Keras. The number of layers can be changed.

hyperparams.py: Hyperparameters. You can change these hyperparams.

preprocessing.py: Preprocessing of the dataset for supervised learning.

FTWS_train_n_days_prediction.py: Prediction of n days of FTWS using n days of climate data, which can be defined in this file by the user.

FTWS_test_n_days_prediction.py: Evaluate the model trained by FTWS_train_n_days_prediction.py.

FTWS_train_one_days_prediction.py: prediction of one day FTWS using n days of climate data.

FTWS_test.py: Evaluate the model trained by FTWS_train_one_days_prediction.py.

Eto_training.py: Predict ETo for one day using n days of climate data.

swc1_diario_biodagro-1652176316.h5 : A trained model with the input of 7 days of the month of the year, average relative humidity, precipitation (mm)',
 ET0 and the output of FTSW on the 7th day. The model  reached  R_square=0.873
and Root mean square of 11.96

swc1_diario_biodagro-1652108950.h5: A trained model with the input of 7 days of the month of the year, tempereture (avg,max,min),
relative humidity (avg,max, min) and the output of ETo on the 7th day. The model  reached  R_square=0.90
and Root mean square of 0.53

