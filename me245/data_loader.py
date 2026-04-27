import pandas as pd
from pathlib import Path
import yfinance as yf
import time 
import shutil
import sys
class DataLoader:
    """
    Handles loading, metadata extraction, and structural validation =,
    of financial datasets.
    
    This class handles the basic data loading for all files to be read from the given file location from the user. It is effectively data preprocessing 1 
    as it is suppose to check if the files are valid and can be preprocessed by the ML project before it goes onto the stage 2 of data preprocessing.
    """
    def load_and_validate_dataset(self):
        print("="*60)
        #convert the input string input into a path object        
        datasets_directory = Path('me245/CSV_Files_training_unverified')
        datasets_paths_verified = []
        
        #now we must go through each file within the directory within the CSV_Files_training_unverified
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        if all_sub_directories != []:
            #reading in all required files in the unverified folder to see if they are usable
            
            print("Available CSV files training unverified to read from:\n")
            for file_path in all_sub_directories:
                #removes the file type at the end of the name. eg pdf for example
                file_name = file_path.stem
                if file_path.suffix == ".csv":
                    print(f"File name: {file_name}  File size: {file_path.stat().st_size / 1024:.2f}Kb   File type: {file_path.suffix}.\n")
                    if self.file_validation(file_path):
                        #you would want to store all the paths that are valid for training and prediction to be in the verified folder
                        print(f"The file {file_name} is a valid file for the models training process.\n")
                        datasets_paths_verified.append(file_path)
                        
                    else:
                        print(f"The file {file_name} is not a valid file please make the changes to the file as it is your own custom one.\n")
            
            #once all the files are checked then we need to be able to decide how many of them will be split up for training. We should simply decided how many files are delegated to training
        else:
            #check if a folder exists and creates it if it doesn't exist
            datasets_directory.mkdir(parents=True,exist_ok=True) 
            print("Directory: me245/CSV_Files_training_unverified Please enter your files into there ")
            sys.exit()
            
        #check if there are enough files that have been accepted 10 is placebo
        if len(datasets_paths_verified) >= 10:
            #we must make the directory if it doesn't exist to store the verified files
            destination_directory = Path("me245/CSV_Files_training_verified")
            destination_directory.mkdir(parents=True,exist_ok=True)
            try:
                #copy the files that are valid into the folder which is verified
                for file in datasets_paths_verified:
                    #need to copy each valid file from the unverified
                    print()
                    shutil.move(file, destination_directory / file.name)
            except Exception as e:
                raise Exception(f"The following problem has occured: {e}.")
        else:
            raise Exception("There aren't enough valid files for the model to be trained on (it must be at least 10).\n")
        
    def file_validation(self,file_path):
        #this checks if the path to the file actually exists
        if not file_path.exists():
            raise ValueError(f"Error: the file '{file_path.name}' doesn't exist.")
        
        try:
            df_dataset = pd.read_csv(file_path)
            #we want the same format for all column headings and removing unecessary whitespaces if there are any
            df_dataset.columns = [''.join(column.lower().split()) for column in df_dataset.columns]
            
            #we need to check that all the columns contain the requried values within this format
            #column headings should be designated as open, high, low, close, volume, adjclose, and date get from yahoo finance
            required_column_names = ["date", "high", "low", "volume", "open", "close", "adjclose"]
            
            #checks against required column names against actual column names
            if not all(col in df_dataset.columns for col in required_column_names):
                print(f"Critical Error: Missing required OHLCV columns in {file_path.name}. They are required to be: 'date', 'high', 'low', 'volume', 'open', 'close', 'adjclose'. Within their respective columns.")
                return False
            
            #if the file needs a dataset above a threshold of required design. For a dataset to be effectively trained on. in this case if less than 1000 days then it won't accept it
            if df_dataset.shape[0] < 1000:
                print(f"The file {file_path.name} does not meet the minimum requirements for the amount of data to be used within this .")
                return False
            
            #since the file meets all required standards and has been checked to be a csv before being entered into this algorithm we will make sure it is ordered by date if it is a custom file
            #convert date string to a day time object and drop duplicate values
            df_dataset['date'] = pd.to_datetime(df_dataset['date'])
            df_dataset = df_dataset.drop_duplicates(subset=['date'], keep='first')
            df_dataset = df_dataset.sort_values(by='date',ascending=True)
            
            #deletes rows with missing values
            df_dataset = df_dataset.dropna().reset_index(drop=True)
            #saves the ordering to the file in a csv
            df_dataset.to_csv(file_path, index=False)
            return True
        except Exception as file_error:
            raise Exception(f"System could not read the file {file_path.name} as a df: {file_error}")
        
        #cannot handle missing data as that would leave too many problems. We must have the assumption that the data is complete like when using yfinance
        #sort the dates within the file then remove the column before fitting as it can lead to overfitting.
        
        
    
    def create_stock_market_dataset(self):
        #WARNING DO NOT USE SESSION FROM IMPORT REQUEST OTHERWISE THIS WILL NOT WORK YFINANCE HAS A IMBUILT SYSTEM RELYING ON curl_cffi backend
        #to reduce throttling of the print function break it down into sub print statements. As it can lead to lines disappearing due to too much concaternation.
        print("="*60 + '\n')
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
                        
                        if len(data) > 0:
                            print(data)
                            #Flatten the MultiIndex headers as they are tuples e.g.('adjclose','comapanyname')
                            data = data.reset_index()
                            
                            #STANDARDIZE THE COLUMNS (Lowercase and remove spaces for validation logic) which moves date as a column instead of index
                            data.columns = [col[0].lower().replace(' ', '') for col in data.columns]
                        
                            print(f"The data that has been requested has been successfully retrieved! {len(data)} rows in the dataset!")
                            
                            
                            '''
                            parent checks if there is a missing file in the given path and if there is one creates it -it creates the parent chain of the file directory given.
                            exist_ok checks if the folder exists if it does then it moves onto the next line of code. It ensure that the code doesn't crash if it exists
                            '''
                            
                            while(True):
                                attempt_answer = input(f"Would you like {name_of_market} to be in the [u]nseen directory or the [t]raining unverified folder? \n")
                                if attempt_answer.lower() == 'u':
                                    directory = Path("me245/CSV_Files_unseen_dataset")
                                    directory.mkdir(parents=True,exist_ok=True) 
                                    break
                                elif attempt_answer.lower() == 't':
                                    #check if a folder exists and define it
                                    directory = Path("me245/CSV_Files_training_unverified")
                                    directory.mkdir(parents=True,exist_ok=True) 
                                    break
                                else:
                                    print("[u]nseen directory or the [t]raining unverified are the only valid inputs please try again.)\n")
                        
                            
                            #this line adds the designated file path to the path class the '/' gets overloaded and is now a function that adds it to the path using the correct slash for the operating system
                            directory = directory / f"{name_of_market}_{start_date_dataset}_{end_date_dataset}.csv"
                            data.to_csv(directory, index=False, encoding="utf-8")
                            
                    #need to wait frequent requests may cause it to stop and deny further access
                    #if there is a problem in access for developers or markers please delete the associated cookies to yfinance to reset this problem
                    time.sleep(5)
                    
                
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
            #to catch network errors
            raise Exception(f"Session with Yahoo finance could not be created due to {WANerror}.")
        finally:
            print("="*60)
    
    #proceedure
    def reset_file_validation(self):
        #move all files from validation on boot up
        datasets_directory = Path('me245/CSV_Files_training_verified')
        datasets_directory.mkdir(parents=True,exist_ok=True)
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        
        #now reuse the same variable as it isn't needed again
        datasets_directory = Path('me245/CSV_Files_training_unverified')
        datasets_directory.mkdir(parents=True,exist_ok=True)
        try:
            #copy the files that are .csv into the folder which is unverified
            for file in all_sub_directories:
                #need to copy each file from the verified folder
                print()
                shutil.move(file, datasets_directory / file.name)
        except Exception as e:
            raise ValueError(f"The following problem has occured: {e}.")