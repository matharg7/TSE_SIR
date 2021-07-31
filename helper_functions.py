import numpy as np
import pandas as pd
import tensorflow as tf

def preprocess(dataframe_csvpath, cols_x, cols_y, window_in, window_out, data_div_frac, popu_size):

    """
    Converts the Csv file into required data format for Time Series prediction    
    
    Arguments:
    dataframe_csvpath -- path of csv file, it has the data for different time sceries
    cols_x -- list of columns to be considered as input to the model(Include 'Series_No' as last entry in list)
    cols_y -- list of columns to be outputed by the model(Include 'Series_No' as last entry in list)
    window_in -- the number of time steps as input
    window_out -- the number of time steps to be predicted
    data_div_frac -- the % of data to be divide into test and train sets  
    popu_size -- population size to normalize data
        
        
    Returns:  
    x_train -- the training input data of shape (m, window_in, len(cols_x)); m is number of examples 
    y_train -- the training data labels for input data of shape (m, window_out, len(cols_y)); m is number of examples 
    x_test -- the testing input data of shape (m, window_in, len(cols_x)); m is number of examples  
    y_test -- the testing data labels of shape (m, window_out, cols_y); m is number of examples  
    len_ser -- the total number of days in a series
    win_len_per_ser -- the total number of windows in a series
    
    """
  
    #Loading .CSV file and creating dataframe
    df = pd.read_csv(dataframe_csvpath)   
    len_ser = len(df[df['Series_No'] == 1])

    #randomly shuffle different series
    permute = np.random.permutation(range(1, len(set(df['Series_No']))))
    train_series_seq = permute[: int(len(set(df['Series_No'])) * data_div_frac)]
    test_series_seq = permute[int( len(set(df['Series_No'])) * data_div_frac):]
    
    #taking relevent columns from dataframe  
    df_x = df[cols_x]
    df_y = df[cols_y]
    
    #Innitialize empty lists which are later to be appended
    x_train, y_train, x_test, y_test = [], [], [], []
    
    #Creating time series data
    for series_no in train_series_seq:
        
        #new dataframe variable assignment for particular series drom df_x, df_y
        series_df_x = df_x[df_x['Series_No'] == series_no]
        series_df_y = df_x[df_y['Series_No'] == series_no]
        
        #converting into numpy arrays
        array_x = np.array(series_df_x)
        array_y = np.array(series_df_y)
        
        #for loop to append to x_train y_train arrays according to window_in, window_out
        for idx in range(len(series_df_x) - window_in - window_out + 1): #'len(series_df_x) - window_in - window_out + 1' needs to be checked
            x_train.append(array_x[idx:idx + window_in, : len(cols_x) - 1]) #out col_x and col_y has last item 'Series number' so to remove that [, : len(cols_x)]
            y_train.append(array_y[idx + window_in :idx + window_in + window_out, : len(cols_y) - 1])

    #repeat for test sequence
    for series_no in test_series_seq:
        
        #new dataframe variable assignment for particular series drom df_x, df_y
        series_df_x = df_x[df_x['Series_No'] == series_no]
        series_df_y = df_x[df_y['Series_No'] == series_no]
        
        #converting into numpy arrays
        array_x = np.array(series_df_x)
        array_y = np.array(series_df_y)
        
        #for loop to append to x_train y_train arrays according to window_in, window_out
        for idx in range(len(series_df_x) - window_in - window_out + 1): #'len(series_df_x) - window_in - window_out + 1' needs to be checked
            x_test.append(array_x[idx:idx + window_in, : len(cols_x) - 1]) #out col_x and col_y has last item 'Series number' so to remove that [, : len(cols_x)]
            y_test.append(array_y[idx + window_in :idx + window_in + window_out, : len(cols_y) - 1])

    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test) 
    x_train[:,:,0:3] = x_train[:,:,0:3] / popu_size 
    x_test[:,:,0:3] = x_test[:,:,0:3] / popu_size
    y_train = y_train / popu_size 
    y_test =  y_test / popu_size
    win_len_per_ser = len_ser - window_in - window_out + 1
    
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), len_ser, win_len_per_ser


def predictions(x_test, y_test, model, len_ser, win_len_per_ser, window_in = 7, window_out = 1):
    """
    Makes Predictions for time series data using the model trained.   
    
    Arguments:
    x_test, y_test -- Testing data
    model -- Model trained previously
    len_seq -- The length of days in a paricular time-series
    window_in -- The input window size for the model
    window_out -- The predicted window size of model
    
    
    Returns:  
    y_pred -- The predicted windows (for plotting)
    mae -- Mean absolute Error of y_pred and y_true
    
    """
  
    num_win_per_ser = win_len_per_ser   #num windows
    #print(num_win_per_ser)
    y_pred = []
    #y_true = []
    for i in range(0, len(y_test), num_win_per_ser): # i takes index values of first windows of different series
        
        win_start = x_test[i] # saving the first window of each series
        #print('win_start:', win_start)
        CR = win_start[0][3] # saving the CR value for particular series -> to be used for prediction
        #print('CR:', CR)
        win = tf.convert_to_tensor(win_start) # window variable which will be updated for new windows, takes first value as the starting window
        #print(win)
        win = tf.reshape(win, (1, win.shape[0], win.shape[1]))
        #print(win.shape)
        for j in range(num_win_per_ser): # prediction loop
            y_hat = model.predict(win) # predicting values wrt win variable
            #print('y_hat', y_hat)  
            y_pred.append(y_hat[0]) # add the value to y_pred
            #print('y_pred:', y_pred)
            y_hat = tf.concat([y_hat, tf.fill(dims = (window_out, 1), value = CR)], axis = 1) # adding CR value y_hat for furter predictions
            #print('cr added to y_hat', y_hat)
            win = tf.concat([win, tf.expand_dims(y_hat, axis = 0)], axis = 1) # adding our prediction to win
            #print('win', win)
            win = win[:,window_out:,:] # updating win by removing the starting elements
            #print('new_win for next iter', win)

    #print(np.array(y_pred)) 
    y_pred = np.array(y_pred)
    print(y_pred.shape)
    print(y_test.shape)
    
    mae = tf.reduce_sum(tf.keras.metrics.mean_absolute_error(y_pred, y_test))
    print(mae)
    return y_pred, mae



  