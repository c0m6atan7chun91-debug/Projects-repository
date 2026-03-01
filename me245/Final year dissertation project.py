import pandas as pd
import os
import numpy as np
from pathlib import Path
import yfinance as yf
import time
import numpy as np
import shutil
from sklearn.ensemble import IsolationForest
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt # Required to save the SHAP plots as images

#p.s. remember to use camel case instead of PascalCase for java
class DataLoader:
    """
    Handles loading, metadata extraction, and structural validation =,
    of financial datasets.
    
    This class handles the basic data loading for all files to be read from the given file location from the user. It is effectively data preprocessing 1 
    as it is suppose to check if the files are valid and can be preprocessed by the ML project before it goes onto the stage 2 of data preprocessing.
    
    hand in the data_set_path as a Path() type it is professional
    Flexibility: If you want to load 10 different datasets in a loop, you don't want to re-initialize the class every time. 
    You want one "Loader" that you can give different paths to.
    Validation: You can check if the path exists before you even try to open the file.
    """
    def load_and_validate_dataset(self):
        print("="*60)
        #convert the input string input into a path object        
        datasets_directory = Path('me245/CSV_Files_training_unverified')
        datasets_paths_verified = []
        
        #now we must go through each file within the directory within the CSV_Files_training_unverified
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        if all_sub_directories != []:
            #reading in all required files for the 
            print("Available CSV files training unverified to read from:\n")
            for file_path in all_sub_directories:
                #removes the file type at the end of the name. eg pdf for example
                file_name = file_path.stem
                if file_path.suffix == ".csv":
                    print(f"File name: {file_name}  File size: {file_path.stat().st_size / 1024:.2f}Kb   File type: {file_path.suffix}.\n")
                    if self.file_validation(file_path):
                        #you would want to store all the paths that are 
                        print(f"The file {file_name} is a valid file for the models training process.\n")
                        datasets_paths_verified.append(file_path)
                        
                    else:
                        print(f"The file {file_name} is not a valid file please make the changes to the file as it is your own custom one.\n")
            
            #once all the files are checked then we need to be able to decide how many of them will be split up for training. We should simply decided how many files are delegated to training
        else:
            #check if a folder exists and creates it if it doesn't exist
            datasets_directory.mkdir(parents=True,exist_ok=True) 
            raise Exception("There are no CSV files to choose from. Please add a CSV file to be read from into the CSV_Files.")
        
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
                raise f"The following problem has occured: {e}."
        else:
            raise Exception("There aren't enough valid files for the model to be trained on (it must be at least 10).\n")
        
    def file_validation(self,file_path):
        #this checks if the path to the file actually exists
        if not file_path.exists():
            raise f"Error: the file '{file_path.name}' doesn't exist."
        
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
            print(f"System could not read the file {file_path.name} as a df: {file_error}")
            return False
        
        #cannot handle missing data as that would leave too many problems. We must have the assumption that the data is complete like when using yfinance
        #sort the dates within the file then remove the column before fitting as it can lead to overfitting.
        
        
    
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
                            directory = Path("me245/CSV_Files_training_unverified")
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
            raise Exception(f"Session with Yahoo finance could not be created due to {WANerror}.")
        finally:
            print("="*60)
            
class Preprocessor:
    
    def classical_machine_learning_data_extraction(self, df_data_to_extract):
        df = df_data_to_extract.copy()
        
        #this can indicate sudden stock splits, indicate someone pumping and dumping, or potential insider training  build up if there is a sustained high volume diff
        df['short_mid_diff_volume'] = df['volume'].rolling(6).mean() - df['volume'].rolling(3).mean()
        df['short_long_diff_volume'] = df['volume'].rolling(12).mean() - df['volume'].rolling(3).mean()
        
        #checks the difference between windows to see if there is a crash edge case at the end of the day and sees if it is just a flash crash or a normal crash
        df['short_mid_diff_adjclose'] = df['adjclose'].rolling(6).mean() - df['adjclose'].rolling(3).mean()
        df['short_long_diff_adjclose'] = df['adjclose'].rolling(12).mean() - df['adjclose'].rolling(3).mean()
        
        #the percentage change from the previous day of the current value. for the HOCLV values
        df['log_adjclose_change'] = np.log(df['adjclose'] / df['adjclose'].shift(1))
        df['log_low_change'] = np.log(df['low'] / df['low'].shift(1))
        df['log_high_change'] = np.log(df['high'] / df['high'].shift(1))
        df['log_close_change'] = np.log(df['close'] / df['close'].shift(1))
        df['log_open_change'] = np.log(df['open'] / df['open'].shift(1))
        df['log_volume_change'] = np.log(df['volume'] / df['volume'].shift(1))
        
        #checks the difference between close and open against the open value.
        #shift() returns the df but moves them back by + if you do -1 it moves them down by 1
        df['difference_open_close'] = (df['open'] - df['close'])/df['open']
        df['difference_high_low'] = (df['high'] - df['low'])/df['open']
        #date extraction the isolation forest needs to tell if a number is normal for a time of year (1-12) for months
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].copy().apply(self.extract_season)
        df = df.drop(columns = ['date'])
        
        #we need to drop all data with missing columns
        df = df.dropna(axis =0, how = 'any')#axis (1 is column 0 is row), how refers to what scenario to drop a collumn (all - all are missing and any - any values are missing), and no need to worry about threshold as there are 1000 datapoints per file
        #potential noise from raw values
        df = df.drop(columns=['open', 'high', 'low', 'close', 'adjclose', 'volume'])
        #normalize all scales so that before data concatination all data is on the same scale and can see what is an outlier for the model so no characters must be used in any feature
        df = (df - df.mean()) / df.std()

        return df
    
    def extract_season(self,month):
        #month is a numeriacal value
        if month in [12,1,2]:
            return 1
        elif month in [3,4,5]:
            return 2
        elif month in [6,7,8]:
            return 3
        elif month in [9,10,11]:
            return 4
        
        
class ClassicalModelManager:
    def __init__(self, contamintion_input):
        # Contamination is the expected % of anomalies (e.g., 3%)
        """
        Initialize the Isolation Forest model.
        
        :param n_estimators: Number of base estimators (trees) in the ensemble.
        :param contamination: Proportion of outliers in the data (0 < contamination <= 0.5).
        :param random_state: Random seed for reproducibility.
        :param n_jobs: decide how many cores are being used
        :param max_samples: decides how many data points will be randomly sampled per tree to be isolated
        :param warmstart: decides if it will continue based off of the previously made trees
        :param bootstrap: It decides if each tree uses a random sample with or without replacement (the latter by default)
        """
        self.isolation_forest = IsolationForest(n_estimators = 200, contamination=contamintion_input,n_jobs=-1)
    
    def classical_model_training(self,train):
        #Each time .fit() is called completely overwrites it previous training each time it is called so it needs to train all in one go
        self.isolation_forest.fit(train) #O(nlogn) time complexity for training
    
    def classical_model_prediction(self,unseen_data):
        #For each observation, tells whether or not (+1 or -1) it should be considered as an inlier according to the fitted model.
        scores = self.isolation_forest.predict(unseen_data)
        #The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.
        prediction = self.isolation_forest.decision_function(unseen_data)
        return scores, prediction
class Interpretation:
    def interpretation_collection(self, scores, predictions,data_extracted, path_name):
        self.anomaly_score_histogram( scores, predictions, path_name)
        self.representation_classical_model( scores, predictions,data_extracted, path_name)
        
    def representation_classical_model(self, scores, predictions,data_extracted, path_name):
       
        # Add scores and labels to dataframe
        data_extracted['anomaly_score'] = predictions
        data_extracted['is_anomaly'] = scores
        filtered_anomaly = data_extracted[data_extracted['is_anomaly'] == -1] #this line also keeps the corresponding rows index after the filter to make sure it matches with the dataextracted df
        #subplots() returns two thing figure-the overall container and axes - the actual plot area where you draw things
        figure, axis = plt.subplots(figsize=(14,5)) # width then height
        #scatter call for normal points
        axis.scatter(data_extracted.index, data_extracted['anomaly_score'],color = 'steelblue',alpha = 1)#x-axis then y-axis and alpha is the transparency
        #scatter call for anomaly points
        axis.scatter(filtered_anomaly.index, filtered_anomaly['anomaly_score'],color = 'red',alpha = 1)#normal is +1 and an anomaly is -1
        axis.axhline(y = 0,color = 'black', linewidth=1) # draws a horizontal line across the entire plot at a given y value. y being where the line sits(threshold),color of line, linewidth,linestyle,label - text shown for legend
        axis.set_xlabel("anomaly index")#xaxis label, font size, font weight, font colour
        axis.set_ylabel("anomaly score")
        plt.title("Diagram of the varying anomaly scores against index occurance")
        plt.tight_layout()#Prevents lables from being cut off
        plt.savefig(path_name)#save figure to a file 
        
        

    def anomaly_score_histogram(self,scores,predictions,path_name):
        normal_scores = predictions[scores == 1]
        anomaly_scores = predictions[scores == -1] #has to be -1 as thats the designation for anomalies
        plt.figure(figsize=(14,5))
        plt.hist([normal_scores,anomaly_scores],label = ["normal_data", "anomaly data"], color = ['skyblue', 'salmon'], alpha = 0.6)#datasets, bin -column width, labels(tuple or list), colour,edgecolor, alpha - transparancy
        plt.axvline(x=0,color='black',linewidth = 1)#draws a verticle line to seperate the two datatypes for easier readability
        plt.title("Histogram of normal data and anomaly data scores")
        plt.ylabel("frequency")
        plt.xlabel("anomaly_score")
        plt.legend()
        plt.tight_layout()#Prevents lables from being cut off
        plt.savefig(path_name)#save figure to a file 
        
        
    def shap_and_lime_diagrams(self):
        fgfg
    
""" 
This class will control the classes and the different sections of the coding project and act as the UI control hub for the user interaction for the project. This is so it can be easily
manage and control the necessary logic for this project between the different subset of
"""
class MainControllerUI:
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.classical_model = ClassicalModelManager(0.01)
        self.interprebility = Interpretation()
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
        
    def start_classical_model(self):
        self.loader.load_and_validate_dataset()
        #convert the input string input into a path object        
        datasets_directory = Path('me245/CSV_Files_training_verified')
        datasets_directory_unverfied = Path('me245/CSV_Files_training_unverified')
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        #for data extraction and model training
        if all_sub_directories != []:
            print("="*60)
            print("\n")
            print("Training has started of the Isolation Forest model. Please wait...\n")
            #we need to recheck the files as the user might have moved them to the wrong location
            dataset_list = []
            for file_path in all_sub_directories:
                if self.loader.file_validation(file_path):
                    data_collected = pd.read_csv(file_path)
                    dataset_list.append(self.preprocessor.classical_machine_learning_data_extraction(data_collected))
                else:
                    shutil.move(file_path, datasets_directory_unverfied / file_path.name)
                    raise f"The file {file_path.stem} doesn't meet the requirements. It has been moved to the folder unverified"
            complete_set_of_training_data = pd.concat(dataset_list, ignore_index=True)
            self.classical_model.classical_model_training(complete_set_of_training_data)
            print("Training of the Isolation Forest model has been completed and predictions are now available\n")
            print("="*60)
            print("\n")
        else:
            raise "no files exist in the verified folder"
        #the new directory we are now working with is the unseen one and we need to create a prediction for the one the user selects
        datasets_directory = Path("me245/CSV_Files_unseen_dataset")
        datasets_directory.mkdir(parents=True,exist_ok=True)
        #need to collect all subdirectories again of files to present to user for selection
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        prediction_outcome = Path("me245/File_of_outcomes")
        prediction_outcome.mkdir(parents=True, exist_ok=True)
        user_wants_predictions = True
        while(user_wants_predictions):
            if all_sub_directories != []:
                print("="*60)
                print("\n")
                print("Please select a file that you want to prediction for:")
                for i,file_path in enumerate(all_sub_directories):
                    #removes the file type at the end of the name. eg pdf for example
                    if file_path.suffix == ".csv":
                        print(f"Index: {i} File name: {file_path.stem}  File size: {file_path.stat().st_size / 1024:.2f}Kb   File type: {file_path.suffix}.\n")
                while(True):
                    try:
                        user_index = int(input("Please enter the index of which file you would like to have predicted out of the ones allocated.\n"))
                        if 0 <= user_index < len(all_sub_directories):
                            break
                    except:
                        print("Please enter a valid input for the input\n")
                file_selected = all_sub_directories[user_index]
                if self.loader.file_validation(file_selected):
                    print("The file is valid to make prediction\n")
                    file_to_predict = pd.read_csv(file_selected)
                    extracted_predict_data = self.preprocessor.classical_machine_learning_data_extraction(file_to_predict)
                    scores, prediction = self.classical_model.classical_model_prediction(extracted_predict_data)
                    self.interprebility.interpretation_collection(scores,prediction,extracted_predict_data ,prediction_outcome / file_selected.stem  )
                    #insert interpretation of prediction function here
                    print("The files' prediction has been made. Please check the prediction folder for the new prediction.\n")
                    
                #check if the user wants to make another prediction using the code below
                while(True):
                    attempt_answer = input("Would you like to make another prediction?([y]es or [n]o).\n")
                    if attempt_answer.lower() == 'n':
                        user_wants_predictions = False
                        break
                    elif attempt_answer.lower() == 'y':
                        break
                    else:
                        print("[y]es or [n]o are the only valid inputs please try again).\n")
                print("="*60)
                print("\n") 
            else:
                print("There are no files in the CSV_Files_unseen_dataset folder. Please enter a file in there that you want to predict before starting the model.\n")
                print("="*60)
                print("\n")
                break
        
        
ui =  MainControllerUI()
ui.copyright_disclaimer()
ui.start()
ui.start_classical_model()