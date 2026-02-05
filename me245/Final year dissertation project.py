import pandas as pd
import os
import numpy as np
from pathlib import Path
import yfinance as yf
import time
import numpy as np
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
    
    
    """
    hand in the data_set_path as a Path() type it is professional
    Flexibility: If you want to load 10 different datasets in a loop, you don't want to re-initialize the class every time. 
    You want one "Loader" that you can give different paths to.
    Validation: You can check if the path exists before you even try to open the file.
    """
    def load_and_validate_dataset(self):
        print("="*60)
        #convert the input string input into a path object        
        datasets_directory = Path('me245/CSV_Files')
        datasets_to_access = []
        datasets_paths_verified = []
        
        #now we must go through each file within the directory within the CSV_Files
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        if all_sub_directories is not None:
            print("Available CSV files to read from:\n")
            #glob
            
            #change logic
            for index,file_path in all_sub_directories:
                #removes the file type at the end of the name. eg pdf for example
                if file_path.suffix == ".csv":
                    print(f"({index})File name: {file_path.stem}  File size: {file_path.stat().st_size / 1024}Kb   File type: {file_path.suffix}.\n")
                    datasets_to_access.append(file_path)
            
            #now we want the users to select which datasets that they would want to use with the model
            while True:
                files_users_chosen = input(f"Please enter a numbers from 0 to {len(all_sub_directories) - 1} for what files you want. e.g 0,2 or 0,1,3,4,5.\n")
                
                try:
                    indices = [int(position) for position in files_users_chosen.split(',')]
                    
                    #first of all check if the indicies are within range
                    if (all(0 <= index <= len(datasets_to_access)) for index in indices):
                        for index in indices:
                            file_to_verify = datasets_to_access[index]
                            if self.file_validation(file_to_verify):
                                #you would want to store all the paths that are 
                                datasets_paths_verified.append(file_to_verify)
                               
                            else:
                                print(f"The file {datasets_to_access[index].stem} is not a valid file please make the changes to the file as it is your own custom one.\n")
                        
                except:
                    return 4
        else:
            #check if a folder exists and creates it if it doesn't exist
            directory = Path("me245/CSV_Files_unverified")
            directory.mkdir(parents=True,exist_ok=True) 
            return "There are no CSV files to choose from. Please add a CSV file to be read from into the CSV_Files."
            
        
    def file_validation(self,dataset_directory):
        #this checks if the path to the file actually exists
        if not dataset_directory.exists():
            raise f"Error: the file '{dataset_directory.name}' doesn't exist."
        
        #this is a format check
        if dataset_directory.suffix != '.csv':
            raise "Error: the file type must be a .csv ."
        
        try:
            df_dataset = pd.read_csv(dataset_directory)
        except Exception as file_error:
            raise f"System could not read the file: {file_error}"
        
        #we want the same format for all column headings and removing unecessary whitespaces if there are any
        df_dataset.columns = [''.join(column.lower().split()) for column in df_dataset.columns]
        
        #we need to check that all the columns contain the requried values within this format
        #column headings should be designated as open, high, low, close, volume, adjclose, and date get from yahoo finance
        required_column_names = ["date", "high", "low", "volume", "open", "close", "adjclose"]
        
        #checks against required column names against actual column names
        if not all(col in df_dataset.columns for col in required_column_names):
            raise ValueError("Critical Error: Missing required OHLCV columns. They are required to be: 'date', 'high', 'low', 'volume', 'open', 'close', 'adjclose'. Within their respective columns.")
        
        #if the file needs a dataset above a threshold of required design. For a dataset to be effectively trained on. in this case if less than 100 days then it won't accept it
        if df_dataset.shape[0] < 100:
            raise "The file does not meet the minimum requirements for the amount of data to be used within this ."
        
        #cannot handle missing data as that would leave too many problems. We must have the assumption that the data is complete like when using yfinance
        
        
    
    def create_stock_market_dataset(self):
        #WARNING DO NOT USE SESSION FROM IMPORT REQUEST OTHERWISE THIS WILL NOT WORK YFINANCE HAS A IMBUILT SYSTEM RELYING ON curl_cffi backend
        #to reduce throttling of the print function break it down into sub print statements. As it can lead to lines disappearing due to too much concaternation.
        print("="*60)
        print("SYSTEM NOTICE: Yahoo Finance API Ingestion Engine")
        print("-" * 60)
        print("1. RATE LIMITS: Do not request the same ticker twice per minute.")
        print("2. THROTTLING: A 2-second buffer is active between downloads.")
        print("3. CACHING: Previously downloaded data is loaded from local CSV.")
        print("="*60)
        try:
            stop_data_collection = False
            while not stop_data_collection:
                name_of_markets_string = input("Enter the name of the markets (capitalized where needed and enter a space between the name the markets if you want more than one).\n")
                start_date_dataset = input("Enter the start date of the dataset (format:YYYY-MM-DD).\n")
                end_date_dataset = input("Enter the end date of the dataset (format:YYYY-MM-DD).\n")
                
                #need to split up the string of markets into a list of markets and removes the spaces between the market names
                list_of_market_names= name_of_markets_string.upper().split()
                
                try:
                    '''
                    we need to go through each individual market as we don't want to lose the data we have collected if it fails
                    if we trip the network limit it cuts off all the data it has collected so it is safer to do them individually
                    It also sends a single index DF
                    '''
                    for name_of_market in list_of_market_names:
                        data = yf.download(name_of_market,start_date_dataset,end_date_dataset,auto_adjust=False)
                        
                        if not data.empty:
                            print(data)
                            
                            #Flatten the MultiIndex headers as they are tuples e.g.('adjclose','comapanyname')
                            data.columns = [col[0].lower() for col in data.columns]
                            
                            #STANDARDIZE THE COLUMNS (Lowercase and remove spaces for validation logic) which moves date as a column instead of index
                            data.reset_index()
                            data.columns = [str(col).lower().replace(' ', '') for col in data.columns]
                            
                            print(data)
                            # Reset index to get 'Date' as a column so the indexs are numbers not the actual dates themselves
                            if 'date' not in data.columns:
                                data = data.reset_index()
                                # Standardize again because 'Date' was just added
                                data.columns = [str(col).lower().replace(' ', '') for col in data.columns]
                        
                            print(f"Success! Captured {len(data)} rows.")
                            
                            #check if a folder exists and define it
                            directory = Path("me245/CSV_Files_unverified")
                            directory.mkdir(parents=True,exist_ok=True) 
                            '''
                            parent checks if there is a missing file in the given path and if there is one creates it -it creates the parent chain of the file directory given.
                            exist_ok checks if the folder exists if it does then it moves onto the next line of code. It ensure that the code doesn't crash if it exists
                            '''
                            
                            #this line adds the designated file path to the path class the '/' gets overloaded and is now a function that adds it to the path using the correct slash for the operating system
                            directory = directory / f"{name_of_market}_{start_date_dataset}_{end_date_dataset}.csv"
                            data.to_csv(directory, index=False, encoding="utf-8")
                            
                    #need to wait frequent requests may cause it to stop and deny further access
                    #if there is a problem in access for developers or markers please delete the associated cookies to yfinance to reset this problem
                    time.sleep(2)
                            
                
                    #User may need more than one dataset
                    while True:
                        attempt_answer = input("Dataset has been created. Would you like to try again? ([y]es or [n]o)\n")
                        if attempt_answer.lower() == 'n':
                            stop_data_collection = True
                            break
                        elif attempt_answer.lower() == 'y':
                            #stops the function and reaches finally
                            break
                        else:
                            print("[y]es or [n]o are the only valid inputs please try again.)\n")
                        
                except Exception as e:
                    while True:
                        print(f"{e}")
                        attempt_answer = input("Data doesn't exist or too many attempts have been made to request it. Would you like to try again? ([y]es or [n]o)\n")
                        if attempt_answer.lower() == 'n':
                            stop_data_collection = True
                            break
                        elif attempt_answer.lower() == 'y':
                            break
                        else:
                            print("[y]es or [n]o are the only valid inputs please try again.)\n")
        except Exception as WANerror:
            raise Exception(f"Session with Yahoo finance could not be created due to {WANerror}.")
        finally:
            print("="*60)
            
""" 
This class will control the classes and the different sections of the coding project and act as the UI control hub for the user interaction for the project. This is so it can be easily
manage and control the necessary logic for this project between the different subset of
"""
class MainControllerUI:
    def __init__(self):
        self.loader = DataLoader()
        self.data_loaded = []
    def start(self):
        return self.loader.create_stock_market_dataset()
    
    def copyright_disclaimer(self):
        print("=" * 60)
        print("Welcome to Anomaly Detection in Financial Market Data!")
        print("Also known as: A.D.F.M.D.")
        print("=" * 60)
        print("\n") # Adds a single clean gap after the banner
        print("="*60)
        #need to enter copyright disclaimer for legal reasons.
        print("Please note: this program uses the yFinance app to extract data using a Yahoo API.\nyFinance is not affiliated to Yahoo, which has it’s own terms relating to reuse of data.\nFor further information please see the yFinance Project Information Page (https://pypi.org/project/yfinance/) and terms of service for Yahoo and Yahoo API’s, both of which can be accessed from the Yahoo Terms page: https://policies.yahoo.com/us/en/yahoo/terms/index.htm.")
        print("="*60)
        print("\n")
        
        
        
MainControllerUI().copyright_disclaimer()
MainControllerUI().start()