import pandas as pd
import os
import numpy as np
from pathlib import Path
#p.s. remember to use camel case instead of PascalCase for java

    

class DataLoader:
    """
    Handles loading, metadata extraction, and structural validation 
    of financial datasets.
    """
    """
    This class handles the basic data loading for all files to be read from the given file location from the user. It is effectively data preprocessing 1 
    as it is suppose to check if the files are valid and can be preprocessed by the ML project before it goes onto the stage 2 of data preprocessing.
    """
    
    def __init__(self):
        self.supported_extentions = ['.csv']
    """
    hand in the data_set_path as a Path() type it is professional
    Flexibility: If you want to load 10 different datasets in a loop, you don't want to re-initialize the class every time. 
    You want one "Loader" that you can give different paths to.
    Validation: You can check if the path exists before you even try to open the file.
    """
    def load_and_validate_dataset(self, dataset_path_string):
        #convert the input string input into a path object        
        dataset_path_object = Path(dataset_path_string)
        
        #this checks if the path to the file actually exists
        if not dataset_path_object.exists():
            return f"Error: the file '{dataset_path_object.name}' doesn't exist."
        
        #this is a format check
        if dataset_path_object.suffix != '.csv':
            return "Error: the file type must be a .csv ."
        
        try:
            df_dataset = pd.read_csv(dataset_path_object)
        except Exception as file_error:
            return f"System could not read the file: {file_error}"
        
        #we want the same format for all column headings and removing unecessary whitespaces if there are any
        df_dataset.columns = [''.join(column.lower().split()) for column in df_dataset.columns]
        
        #we need to check that all the columns contain the requried values within this format
        #column headings should be designated as open, high, low, close, volume, adjclose, and date get from yahoo finance
        required_column_names = ["date", "high", "low", "volume", "open", "close", "adjclose"]
        
        #checks against required column names against actual column names
        contain_all_required_titles = [column for column in df_dataset.columns if column in required_column_names]
        
        if not contain_all_required_titles:
            return "The file has not named its columns correctly. They are required to be: 'date', 'high', 'low', 'volume', 'open', 'close', 'adjclose'. Within their respective columns."
        
        #if the file needs a dataset above a threshold of required design. For a dataset to be effectively trained on. in this case if less than 10 lines then it won't accept it
        if df_dataset.shape[0] < 100:
            return "The file does not meet the minimum requirements for the amount of data to be used within this ."
        
        #now handle missing data as it is acceptable
        self.handle_missing_data(df_dataset)
        
    #checks if the columns have missing values within the dataset and will remove them if necessary
    def handle_missing_data(self, input_df_dataset):
        #input_df_dataset is the df of a dataset that has been accepted for being an acceptable dataset.
        
        #drops rows with missing dates as they are useless
        input_df_dataset = input_df_dataset["date"].dropna()
        
        #extrapolate each columns rows with datasets using the interpolation if there are missing data in columns
        
            
            
        
        
            
        
        
        
        
        
""" 
This class will control the classes and the different sections of the coding project and act as the UI control hub for the user interaction for the project. This is so it can be easily
manage and control the necessary logic for this project between the different subset of
"""
class MainControllerUI:
    loader = DataLoader()
        
        