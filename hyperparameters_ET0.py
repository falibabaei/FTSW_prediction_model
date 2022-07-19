#hyperparameters
seq_len=7 #(number of look backs of the LSTM model)
output_len=2 #(number of days to predict ET)
n_out=1 #(number of variables to predict)
Batch_size=63#50#40 #128
Epoch=500
lr= 0.00089#0.00025856#0.0003968 #2e-3
decay= 0.00094#0.0003011 #0.0003456#1e-3
hidden_units=38 #number of hidden neurons in LSTM layer 32,64,128,256,
dropout_size=0.09313# 0.17521#0.09313
n_layers=2#number of BiLSTM layers in model

