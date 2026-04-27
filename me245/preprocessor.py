import pandas as pd
import numpy as np 
import torch

class Preprocessor:
    
    def machine_learning_data_extraction(self, df_data_to_extract):
        df = df_data_to_extract.copy()
        df = df[['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']] # ensures that it only contains those columns
        #this can indicate sudden stock splits, indicate someone pumping and dumping, or potential insider training  build up if there is a sustained high volume diff
        minimum_rolling_window_volume = df['volume'].rolling(3).mean()
        df['short_mid_diff_volume'] = df['volume'].rolling(6).mean() - minimum_rolling_window_volume
        df['short_long_diff_volume'] = df['volume'].rolling(12).mean() - minimum_rolling_window_volume
        
        #checks the difference between windows to see if there is a crash edge case at the end of the day and sees if it is just a flash crash or a normal crash
        minimum_rolling_window_adjclose = df['adjclose'].rolling(3).mean()
        df['short_mid_diff_adjclose'] = df['adjclose'].rolling(6).mean() - minimum_rolling_window_adjclose
        df['short_long_diff_adjclose'] = df['adjclose'].rolling(12).mean() - minimum_rolling_window_adjclose
        
        #the percentage change from the previous day of the current value. for the HOCLV values
        # .clip(lower=1e-9) it replaces any value when it is below 0 with 0.000000001 to avoid division by 0 errors
        df['log_adjclose_change'] = np.log(df['adjclose'].clip(lower=1e-9) / df['adjclose'].shift(1).clip(lower=1e-9))
        df['log_low_change'] = np.log(df['low'].clip(lower=1e-9) / df['low'].shift(1).clip(lower=1e-9))
        df['log_high_change'] = np.log(df['high'].clip(lower=1e-9) / df['high'].shift(1).clip(lower=1e-9))
        df['log_close_change'] = np.log(df['close'].clip(lower=1e-9) / df['close'].shift(1).clip(lower=1e-9))
        df['log_open_change'] = np.log(df['open'].clip(lower=1e-9) / df['open'].shift(1).clip(lower=1e-9))
        df['log_volume_change'] = np.log(df['volume'].clip(lower=1e-9) / df['volume'].shift(1).clip(lower=1e-9)) # numerators need to be clipped as well to stop -inf errors in the preprocessor
        
        #checks the difference between close and open against the open value.
        #shift() returns the df but moves them back by + if you do -1 it moves them down by 1
        df['difference_open_close'] = (df['open'] - df['close'])/df['open'].clip(lower=1e-9)
        df['difference_high_low'] = (df['high'] - df['low'])/df['open'].clip(lower=1e-9)
        #date extraction the isolation forest needs to tell if a number is normal for a time of year (1-12) for months
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].copy().apply(self._extract_season)
        df = df.set_index('date')
        
        #we need to drop all data with missing columns
        df = df.dropna(axis =0, how = 'any')#axis (1 is column 0 is row), how refers to what scenario to drop a collumn (all - all are missing and any - any values are missing), and no need to worry about threshold as there are 1000 datapoints per file
        #potential noise from raw values
        df = df.drop(columns=['open', 'high', 'low', 'close', 'adjclose', 'volume'])
        #normalize all scales so that before data concatination all data is on the same scale and can see what is an outlier for the model so no characters must be used in any feature
        df = (df - df.mean()) / df.std()
        
        
        return df
    
    def _extract_season(self,month):
        #month is a numeriacal value
        if month in [12,1,2]:
            return 1
        elif month in [3,4,5]:
            return 2
        elif month in [6,7,8]:
            return 3
        elif month in [9,10,11]:
            return 4
        
    def create_sequences(self,dataset_extracted ,window_size = 60):
        #dataset_extracted is a df
        #Each window will be 60 days and a list of individual tensors will store these said windows
        #Also df headers are removed when coverted to numpy so there is no need to worry about removing them. I had to check online.
        #The torch.tensor converts the numpy array (previously a df) to a tensor while maintaining its shape and columns
        #There is no need to worry about having no titles as the extracted datasets are always created in the same order
        list_of_tensor_sequences = []
        for starting_datapoint in range(len(dataset_extracted) - window_size):
            list_of_tensor_sequences.append(torch.tensor(dataset_extracted[starting_datapoint:starting_datapoint + window_size].values).float())# needs .values to extract the values
        tensor_of_combined_sequences = torch.stack(list_of_tensor_sequences) #stack them so that training can go over them in batches
        return tensor_of_combined_sequences