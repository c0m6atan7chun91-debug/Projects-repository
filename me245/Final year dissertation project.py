import pandas as pd
import os
import numpy as np
from pathlib import Path
import yfinance as yf
import time
import numpy as np
import shutil
from sklearn.ensemble import IsolationForest

#p.s. remember to use camel case instead of PascalCase for java
class DataLoader:
    """
    Handles loading, metadata extraction, and structural validation =,
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
        datasets_directory = Path('me245/CSV_Files_training_unverified')
        datasets_paths_verified = []
        
        #now we must go through each file within the directory within the CSV_Files_training_unverified
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        if all_sub_directories is not None:
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
            df_dataset.to_csv(file_path)
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
            
class Preprocessor:
    
    def classical_machine_learning_data_extraction1(self, df_data_to_extract):
        df = df_data_to_extract.copy()

        # (high - low) / open — intraday movement relative to the opening price.
        # Anomalies detected: flash crashes, short squeezes, and market manipulation
        # (e.g. spoofing). These events produce extreme intraday swings even when
        # the closing price appears superficially normal in isolation.
        df['intraday_range'] = (df['high'] - df['low']) / df['open']

        # Current volume divided by its 20-day rolling mean.
        # Anomalies detected: insider trading (unusual volume before announcements),
        # pump-and-dump schemes, panic selling/buying events, and earnings surprises.
        # A ratio of 5x is far more informative than the raw volume figure alone.
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # (open_t - close_{t-1}) / close_{t-1} — the overnight price shock.
        # Anomalies detected: earnings shocks, geopolitical events, regulatory
        # announcements, M&A news, and black swan events. These information-driven
        # gaps do not appear in intraday data and are a classic anomaly signature.
        df['close_to_open_gap'] = (df['open'] - df['adjclose'].shift(1)) / df['adjclose'].shift(1)


        """Candlestick Wick Ratios (Upper & Lower Shadow)
        Upper shadow: how far the price was pushed above the open/close (based on which is higher) then is divided by the difference of the stock from its lowest compared to its highest.
        Lower shadow: how far the price was pushed below the open/close (based on which is lower) then is divided by the difference of the stock from its lowest compared to its highest.
        A large upper wick means buyers drove the price up but sellers forcefully sell leading to it going significantly lower comapared to a high point
        A large lower wick means that a buyer sold his shares causing 
        Unlike intraday_range which only captures total width, these capture where the
        close landed within that range, revealing the directional intent behind the move.
        Anomalies detected: failed pump-and-dump attempts  (which can cause a flash crash), spoofing(the buyer sells and sells all his shares causing other sellers to sell to drive the price down then a buyer
        can buy them at a much lower price to drive the price up(this can cause a flash crash) if there is someone who can buy the shares afterwards)
        capitulation events, and institutional accumulation/distribution."""
        intraday_width = df['high'] - df['low']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / intraday_width
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / intraday_width

        """The remaining OHLCV columns still need the same treatment — a raw data rate of change isn't consistent and not comparable across different stocks or price levels.
        log_open_change   — captures sudden day-over-day open return in relaion to the previous day
        log_high_change   — captures sudden day-over-day high return in relaion to the previous day
        log_low_change    — captures sudden day-over-day low return in relaion to the previous day; captures consecutive lower lows signal sustained selling pressure and potential capitulation events.
        log_volume_change — captures sudden day-over-day volume return in relaion to the previous day"""
        df['log_open_change']   = np.log(df['open']   / df['open'].shift(1))
        df['log_high_change']   = np.log(df['high']   / df['high'].shift(1))
        df['log_low_change']    = np.log(df['low']    / df['low'].shift(1))
        df['log_volume_change'] = np.log(df['volume'] / df['volume'].shift(1))
        df['log_adjclose_change'] = np.log(df['adjclose'] / df['adjclose'].shift(1))
        
        """Multi-Scale Rolling Volatility
        Three windows give the model a volatility "regime" so it can distinguish
        between a sustained volatile period and a sudden isolated spike.
        Anomalies detected: systemic risk events (e.g. COVID crash, 2008 crisis), and short-lived shock events.
        A 5-day spike inside a calm 20-day window is far more suspicious than the same spike during an already volatile period.
        We also use use adjclose instead of the other Values due to adjclose is a much better measure than the rest of the raw values.
        As it adjusts for stock splits and dividends; so a 2 to 1 split doesn't show up as a fake drop in your returns. So a split would cause massive outliers in the rest of the raw values in the dataset.
        Also if we did apply it to high or low it would measure the volatility of the values as they can fluctuate highly through out a course of a set time period.
        This helps identify flash crashes as if there is a sharp spike in a 5 day window and the 10 day/20day window doesn't pick it up significantly its then likely to be a flash crash. though it can also find other anomalies such as a gap down event"""
        df['rolling_volatility_5d']  = df['log_adjclose_change'].rolling(window=5).std()
        df['rolling_volatility_10d'] = df['log_adjclose_change'].rolling(window=10).std()
        df['rolling_volatility_20d'] = df['log_adjclose_change'].rolling(window=20).std()


        # Year is kept as a plain integer (not cyclical) — years are linear and do not
        # wrap around. This allows the model to contextualise what is "normal" per year,
        # e.g. a volatility spike normal in 2020 (COVID) may be anomalous in 2023.
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

        # Sine/Cosine transforms for month so the model understands Dec (12) is
        # adjacent to Jan (1) — months are cyclical, years are not.
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

        # Date is not a numeric feature — keeping it risks data leakage
        df = df.drop(columns=['date'])

        # Shift and rolling windows introduce NaN at the head of the dataset.
        # We do not impute — fabricating values would teach the model that
        # stagnant flat periods are normal, reducing anomaly sensitivity.
        df = df.dropna().reset_index(drop=True)

        return df
    
    def data_split_80_20_and_zscore(self,df):
        #the following code creates a 80/20 split
        split_index = int(len(df) * 0.8)
        df_train = df.iloc[:split_index].reset_index(drop=True)
        df_val   = df.iloc[split_index:].reset_index(drop=True)
        
        # (x - mean) / std — centres each feature at 0 with unit variance.
        # This ensures no single attribute dominates the anomaly score due to scale.
        #if i normalize it all in one go that would be causing data leakage as it would go into the validation set
        #The alternative here is to see what is normal behaviour within the market and apply that same logic to the 
        #validation set to identify these anomalies based on normal behaviour
        df_train_mean = df_train.mean()
        df_train_std = df_train.std()
        df_train = (df_train - df_train_mean) / df_train_std
        df_val = (df_val - df_train_mean) / df_train_std
        #returns the splits
        return df_train, df_val
    
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
        self.iso_forest = IsolationForest(n_estimators = 200, contamination=contamintion_input,n_jobs=-1)
        

""" 
This class will control the classes and the different sections of the coding project and act as the UI control hub for the user interaction for the project. This is so it can be easily
manage and control the necessary logic for this project between the different subset of
"""
class MainControllerUI:
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.classical_model = ClassicalModelManager(0.03)
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
        if all_sub_directories is not None:
            #we need to recheck the files as the user might have moved them to the wrong location
            for file_path in all_sub_directories:
                if self.loader.file_validation(file_path):
                    data_extracted = pd.read_csv(file_path)
                    data_extracted = self.preprocessor.classical_machine_learning_data_extraction1(data_extracted)
                    training_data, validation_data = self.preprocessor.data_split_80_20_and_zscore(data_extracted)
                    
                    #we create all the features
                    #do a split and normalize the data based on the normality of the training data
                    #then we starting training the model and then use the validation data for it
                else:
                    shutil.move(file_path, datasets_directory_unverfied / file_path.name)
                    raise f"The file {file_path.stem} doesn't meet the requirements. It has been moved to the folder unverified"
        else:
            raise "no files exist"
        
        
ui =  MainControllerUI()
ui.copyright_disclaimer()
ui.start()
ui.start_classical_model()