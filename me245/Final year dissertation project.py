import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
import time
import numpy as np
import shutil
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import matplotlib.pyplot as plt # Required to save the SHAP plots as images



device = torch.device('cpu') # this line states if the GPU is available and then uses the CUDA which is an API made by NVIDIA that allows the program to use the GPU for computation
#Not a language but uses c/c++ extentions.It uses tensor operations on the GPU. This is because the GPU is designed for parallel computation through many mini cores compared to a CPU few but powerful cores
#Neuron training is basically a neuron that takes in a matrix input and return a singular output into another to eventually create an output
#It is defined up here as multiple classes need it



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
                            data.columns = data[0].str.lower()
                            
                            #STANDARDIZE THE COLUMNS (Lowercase and remove spaces for validation logic) which moves date as a column instead of index
                            data.columns = [str(col).lower().replace(' ', '') for col in data.columns]
                        
                            print(f"The data that has been requested has been successfully retrieved! {len(data)} rows in the dataset!")
                            
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
            raise f"The following problem has occured: {e}."
        
            
class Preprocessor:
    
    def machine_learning_data_extraction(self, df_data_to_extract):
        df = df_data_to_extract.copy()
        
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
        df['log_volume_change'] = np.log(df['volume'].clip(lower=1e-9) / df['volume'].shift(1).clip(lower=1e-9))
        
        #checks the difference between close and open against the open value.
        #shift() returns the df but moves them back by + if you do -1 it moves them down by 1
        df['difference_open_close'] = ((df['open'] - df['close'])/df['open']).clip(lower=1e-9)
        df['difference_high_low'] = ((df['high'] - df['low'])/df['open']).clip(lower=1e-9)
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
    
'''
The autoencoder class was developed using the help of a tutorial: https://www.youtube.com/watch?v=zp8clK9yCro&t=214s and https://www.youtube.com/watch?v=VVDHU_TWwUg
It only was used to establish syntax and theory in regarding the coding side of pytorch. I still had to change this section with my own theory and found syntax to the project. I also applied some syntax it taught in said video to the AutoencoderModelManager class
We want to inherit from the NN class as it gives us access to the Mean Squared error loss function for example so that we don't need to define these functions ourselfs
'''
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__() # required to initialize the parent class that is nn to use its functions
        
        #Define the NN layers for the autoencoder
        '''
        we have 15 features and have 60 data points being entered into the dataset
        So the linear layer also known as the input layer will be condensing our input features into the NN
        60 days to cover a quater of the year and each day has 15 features that has been extracted from it so 60*15 = 900 features to enter the neural network
        
        As there is a tanh function it will only output values between 1 and -1, which is suitable for normalized datasets
        '''
        #Define the encoder:
        self.encoder = nn.Sequential(
            nn.Linear(900,90),#input layer so 900 represents the total features and 90 represents the vector it gets compressed into for the next input space
            nn.Tanh(),#activation function
            nn.Linear(90,45),#hidden layer
            nn.Tanh()#activation function at this point there are 60 samples having its features compressed into 45 values in a input space
            #Don't use the ReLU to allow negative values from the normalization. If I used ReLU function there will be no negative values to return so it will destroy performance.
            #You should use Tanh as it maps the values between -1 and 1 which is slightly smaller scale than a normalized dataset but still would get the model to learn the fundermental patterns. Using a adjusted scale.
        )
        
        #For the decoder it needs to do the reverse operations of the encoder to recreate the dataset
        self.decoder = nn.Sequential(
            nn.Linear(45,90),#hidden layer
            nn.Tanh(),
            nn.Linear(90,900)#output layer and no activation function required as the raw output should be the normalized values limiting it between 1 and -1 would skew the MSEs evaluation of the output
        )
        self.to(device)
        
    def forward(self,input_for_neuron):
        encoded = self.encoder(input_for_neuron)
        decoded = self.decoder(encoded)
        return decoded
    
#I have made the AutoencoderModelManager class myself
#the training_loop was developed using the inpiration from a tutorial and some details are similar: https://www.youtube.com/watch?v=zp8clK9yCro&t=214s
class AutoencoderModelManager():
    def __init__(self):
        self.autoencoder_model = Autoencoder() #Establish the model we use to execute said calculations
        self.training_criterion = nn.MSELoss() #You do this to establish the function that will be used to compare the input and output values of the autoencoder, so it is the criterion
        self.prediction_criterion = nn.MSELoss(reduction='none') #reduction='none' instead of allowing it to return one average loss value for the entire batch entered (like the training one). It returns a tensor of the same shape as the input (meaning each input sequence has an assigned loss value).
        self.optimization = torch.optim.Adam(self.autoencoder_model.parameters(), lr = 0.0001) #this updates the weights based on the learning rate which controls how much these weights are updated by. Also, self.autoencoder_model.parameters() directs which weights to update for the ML algorithm and lr is in reference to lr
        #Adam seems to be the best optimizer based from this source: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
        
    def training_loop(self, sequences):
        #A sequence is how many 60 day sliding windows we can recreate from a given training dataset
        #An epoch is how many times the model goes over a training dataset during training (forward and backwards (backwards is to check the weights and see if they need to be changed))
        for epoch in range(20):
            for batch_increment in range(0, len(sequences), 32):
                #do the forward then backward pass of the loop and update the weights
                batch = sequences[batch_increment : batch_increment+32]
                input_vector_sequence = batch.reshape(-1,900) #doing -1 will infer the 2nd dimension and the 900 will say how many data points are expected per column. as 60*15 is = 900 it will create a vector with 900 elements. So it takes in batches of 32 sequences at a time
                reconstructed_sequence_data = self.autoencoder_model(input_vector_sequence) #enter the required sequence for training into the model to create a model output of it
                loss = self.training_criterion(reconstructed_sequence_data, input_vector_sequence) #you check the difference between the reconstructed and input sequence by using the MSE loss function to find how different it is
                
                self.optimization.zero_grad() #this clears the gradients history that was previously calculated, otherwise you will have incremented the sum of the previous one
                loss.backward()#Go back across the dataset using chain rule to check the derivatives and see how the loss function to see the d loss/ d Weight for the gradient of the loss against the weight function.
                #torch.nn.utils.clip_grad_norm_(self.autoencoder_model.parameters(), max_norm=1.0)# This line ensure that the weights are scaled proportionately to one another avoiding a single weight from being exessively larger than the rest
                self.optimization.step()#this updates the weights of the autoencoder to try and minimize the difference between the output from the autoencoder and the input to the autoencoder. It then uses the gradient as such: weight - lr * gradient.
            print(f"Epoch: {epoch + 1 }, Loss: {loss.item():.4f}") #loss is the difference between the constructed and reconstructed output
    
    #This loop was made entirely without tutorials
    def prediction(self,sequences):
        '''
        To understand what makes an anomaly in this case we will have to look at how the model is trained. It needs to be trained on a large pool of data to understand what is considered a normal dataset.
        So assuming it has learned what is a normal behaviour from that; datasets that are entered should have "normal" periods to the anomaly detection system. So if it cannot be recreated then it must be an extreme outlier compared to what is normal market data.
        So if a sequence is above the mean of the loss and 2 times more than standard deviations summed up in a one tailed test then it is an anomaly.
        It is using a one tailed test as you can't get an error below zero has MSE uses the square difference.
        '''
        list_of_loss_values = [] #append mean value of a sequence and the starting index (which is the ordinal index that is the same to the dfs so i can find the starting date)
        self.autoencoder_model.eval() #this sets the model into evaluation mode. It disables dropout and batch normalization layers
        #Ensures that it stops making a computation graph during a forward pass. So it doesn't take up more memory.
        with torch.no_grad():
            for sequence in sequences:
                input_vector_sequence = sequence.reshape(-1,900) #doing -1 will infer the 2nd dimension and the 900 will say how many data points are expected per column. as 60*15 is = 900 it will create a vector with 900 elements
                reconstructed_sequence_data = self.autoencoder_model(input_vector_sequence) #enter the required sequence for training into the model to create a model output of it
                losses_per_sequence = self.prediction_criterion(reconstructed_sequence_data, input_vector_sequence).mean(dim=1) #you check the difference between the reconstructed and input sequence by using the MSE loss function to find how different it is and .mean() returns each sequences loss values from a batch in a vector of 1 dimension but as an average for that sequence
                for loss in losses_per_sequence:
                    list_of_loss_values.append(loss.item())
                
            #Now we have the list of loss values and each index is corresponding to the date it starts at from the df and dataset it is from
            #convert it to a torch temporarily to a tensor. This is becuase Tensors build off of the numpy extension meaning their sizes are fixed
            tensor_of_loss_values = torch.tensor(list_of_loss_values)
            mean_loss = tensor_of_loss_values.mean() #returns a tensor and in this case is the mean of the mean of sequences
            std_loss = tensor_of_loss_values.std() #return a tensor  so need to use .item() to return the actual std.std reconstruction errors are used to create the threshold to find how many lie in a 99% confidence based on the relative distance from the mean of sequences more or less a span of how much the population should lie in
            #This is where I found the measuremetn of a 99% confidence interval for a one tailed test: https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Statistics_with_Technology_2e_(Kozak)/12%3A_Appendix-_Critical_Value_Tables/12.02%3A_Normal_Critical_Values_for_Confidence_Levels
            #so you use the 98% confidence interval (which is for two tail tests) as we are using one tail it will cover 99% of the population so it is 2.33 std away from the mean
            list_of_loss_values_and_dates = [[value,i] for i, value in enumerate(list_of_loss_values) if value > 2.33* std_loss.item() + mean_loss.item()]
            return list_of_loss_values_and_dates
        
        
                
        
        
class Interpretation:
    def interpretation_collection(self, scores, predictions,data_extracted, path_name, isolation_forest_model):
        #with_name() allows me to change the name of the stem path
        self.anomaly_score_histogram( scores, predictions, path_name / "histogram.png")
        self.representation_classical_model( scores, predictions,data_extracted, path_name / "scatter.png")
        
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
        
        
    def shap_diagrams(self,isolation_forest_model, data_extracted):
        shap.summary_plot()
    
""" 
This class will control the classes and the different sections of the coding project and act as the UI control hub for the user interaction for the project. This is so it can be easily
manage and control the necessary logic for this project between the different subset of
"""
class MainControllerUI:
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.classical_model = ClassicalModelManager(0.01)
        self.deep_learning_model_manager = AutoencoderModelManager()
        
        
        
        
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
                    dataset_list.append(self.preprocessor.machine_learning_data_extraction(data_collected))
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
                #we need to call a directory the file_name to store the out come
                asset_prediction = prediction_outcome / file_selected.stem
                asset_prediction.mkdir(parents=True,exist_ok=True)
                if self.loader.file_validation(file_selected):
                    print("The file is valid to make prediction\n")
                    file_to_predict = pd.read_csv(file_selected)
                    extracted_predict_data = self.preprocessor.machine_learning_data_extraction(file_to_predict)
                    scores, prediction = self.classical_model.classical_model_prediction(extracted_predict_data)
                    self.interprebility.interpretation_collection(scores,prediction,extracted_predict_data ,asset_prediction,self.classical_model)
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
    
    def start_deep_learning_model(self):
        #we need to check if the files are valid again in case of changes
        self.loader.reset_file_validation()
        self.loader.load_and_validate_dataset()#check if the datasets are valid in the unverified folder
        
        
        
        #convert the input string input into a path object        
        datasets_directory = Path('me245/CSV_Files_training_verified')
        datasets_directory_unverfied = Path('me245/CSV_Files_training_unverified')
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        #for data extraction and model training
        if all_sub_directories != []:
            print("="*60)
            print("\n")
            print("Training has started of the autoencoder model. Please wait...\n")
            #we need to recheck the files as the user might have moved them to the wrong location
            #this is where we execute the data
            dataset_list = []
            for file_path in all_sub_directories:
                if self.loader.file_validation(file_path):
                    data_collected = pd.read_csv(file_path)
                    dataset_list.append(self.preprocessor.machine_learning_data_extraction(data_collected))
                    #print(data_extracted.isin([np.inf, -np.inf]).any())
                else:
                    shutil.move(file_path, datasets_directory_unverfied / file_path.name)
                    raise f"The file {file_path.stem} doesn't meet the requirements. It has been moved to the folder unverified"
                
            #here you change the training dataset as you can't just combine them together
            
            
            #loop for deep model training
            #it is slow but due to the size and how small the bottleneck is
            file_being_trained = 0
            for dataset_preprocessed in dataset_list:
                print(f"file being trained: {file_being_trained}")
                file_being_trained+=1
                tensors_of_sequences_batches = self.preprocessor.create_sequences(dataset_preprocessed)
                self.deep_learning_model_manager.training_loop(tensors_of_sequences_batches)
                
                
            
            
            
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
                #we need to call a directory the file_name to store the out come
                asset_prediction = prediction_outcome / file_selected.stem
                asset_prediction.mkdir(parents=True,exist_ok=True)
                if self.loader.file_validation(file_selected):
                    print("The file is valid to make prediction\n")
                    file_to_predict = pd.read_csv(file_selected)
                    
                    
                    
                    #so we need to extract the predicted data and then create the predictions that it recieves
                    
                    
                    
                    extracted_predict_data = self.preprocessor.machine_learning_data_extraction(file_to_predict)
                    tensor_of_combined_sequences = self.preprocessor.create_sequences(extracted_predict_data)
                    list_of_loss_values_and_dates = self.deep_learning_model_manager.prediction(tensor_of_combined_sequences)
                    print(list_of_loss_values_and_dates)
                    print(f"The anomaly rate found in the dataset is {len(list_of_loss_values_and_dates)/len(extracted_predict_data):.4f}%")
                    
                    
                    
                    
                    
                    
                    
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
#ui.start()
ui.start_deep_learning_model()
#ui.start_classical_model()S