import pandas as pd
import numpy as np 
from pathlib import Path
from data_loader import DataLoader
from preprocessor import Preprocessor
from isolation_forest_manager import IsolationForestManager
from autoencoder_model_manager import AutoencoderModelManager
from interpretation import Interpretation
import shutil
from datetime import datetime

""" 
This class will control the classes and the different sections of the coding project and act as the UI control hub for the user interaction for the project. This is so it can be easily
manage and control the necessary logic for this project between the different subset of
"""
class MainControllerUI:
    def __init__(self):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.interprebility = Interpretation()
        self.data_loaded = []

    def retrieve_stock_market_data(self):
        return self.loader.create_stock_market_dataset()
    
    def yes_or_no_selection(self, statement):
        while(True):
            user_input = input(statement)
            if user_input.lower() == "y":
                return True
            elif user_input.lower() == "n":
                return False
            else:
                print("[y]es or [n]o are the only valid inputs please try again).\n")
                
    def model_training_ui(self,model_created):
        #convert the input string input into a path object        
        datasets_directory = Path('CSV_Files_training_verified')
        datasets_directory_unverfied = Path('CSV_Files_training_unverified')
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        #for data extraction and model training
        if all_sub_directories != []:
            print("="*60)
            print("\n")
            print(f"Training has started of the {model_created.model_name} model. Please wait...\n")
            
            #we need to recheck the files as the user might have moved them to the wrong location
            #this is where we execute the data re-validation and preprocessing occurs below
            dataset_list = []
            for file_path in all_sub_directories:
                
                if self.loader.file_validation(file_path):
                    data_collected = pd.read_csv(file_path)
                    dataset_list.append(self.preprocessor.machine_learning_data_extraction(data_collected))
                    
                else:
                    shutil.move(file_path, datasets_directory_unverfied / file_path.name)
                    raise ValueError(f"The file {file_path.stem} doesn't meet the requirements. It has been moved to the folder unverified")
                
            #change instructions based on wanted model for what model needs to be trained
            if(isinstance(model_created,IsolationForestManager)):
                complete_set_of_training_data = pd.concat(dataset_list, ignore_index=True)
                model_created.training_loop(complete_set_of_training_data)
                
            elif(isinstance(model_created, AutoencoderModelManager)):
                #loop for deep model training
                #it is slow but due to the size and how small the bottleneck is
                file_being_trained = 0
                for dataset_preprocessed in dataset_list:
                    print(f"file being trained: {file_being_trained}")
                    file_being_trained+=1
                    tensors_of_sequences_batches = self.preprocessor.create_sequences(dataset_preprocessed)
                    model_created.training_loop(tensors_of_sequences_batches)
                
            print(f"Training of the {model_created.model_name} model has been completed and predictions are now available\n")
            print("="*60)
            print("\n")
            
        else:
            raise ValueError("no files exist in the verified folder")#valueerror() signals that a function has recieved an argement of the right type but is unacceptable
    
    def prediction_ui(self,model_created):
        #the new directory we are now working with is the unseen one and we need to create a prediction for the one the user selects
        datasets_directory = Path("CSV_Files_unseen_dataset")
        datasets_directory.mkdir(parents=True,exist_ok=True)
        #need to collect all subdirectories again of files to present to user for selection
        all_sub_directories = list(datasets_directory.glob("*.csv"))
        prediction_outcome = Path(f"File_of_outcomes/{model_created.model_name}")
        prediction_outcome.mkdir(parents=True, exist_ok=True)
        user_wants_option = True # used to see if the user wants to make a prediction or save model
        
        #loop to make predictions
        while(user_wants_option):
            if all_sub_directories != []:
                print("="*60)
                print("Prediction UI")
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
                    
                    
                    #here we make predictions depending on which model is in use. we apply the model and create an output
                    if(isinstance(model_created,IsolationForestManager)):
                        extracted_predict_data = self.preprocessor.machine_learning_data_extraction(file_to_predict)
                        scores, prediction = model_created.prediction(extracted_predict_data)
                        self.interprebility.interpretation_collection(scores, prediction, extracted_predict_data, asset_prediction, model_created.isolation_forest)
                        
                        
                    elif(isinstance(model_created, AutoencoderModelManager)):
                        extracted_predict_data = self.preprocessor.machine_learning_data_extraction(file_to_predict)
                        tensor_of_combined_sequences = self.preprocessor.create_sequences(extracted_predict_data)
                        list_of_dates = model_created.prediction(tensor_of_combined_sequences)
                        print(f"The anomaly rate found in the dataset is {len(list_of_dates)*100/len(extracted_predict_data):.4f}%")
                        #now we only need to group the values that have consecutive range of 60 between each other. for example 1,23,56,77, 544 would be grouped as [1,77],[544] for matlib
                        #for Shap you need the actual sequence itself
                        splice_indexs = np.where(np.diff(list_of_dates) > 15)[0]+1#np.diff() returns the differences between indexs in the list. Also, [0] returns the actual array of indicies and the + 1 represents where to cut after the selected value so for example you want to cut after 6 the +1 indicates to cut after it
                        list_of_dates = np.split(list_of_dates,splice_indexs)
                        dates_not_filtered = pd.to_datetime(file_to_predict['date']).isin(extracted_predict_data.index)#check for dates that haven't been filtered during extraction
                        raw_filtered = file_to_predict[dates_not_filtered].reset_index(drop=True) #only accept data that has the dates that were extracted
                        self.interprebility.anomaly_autocorrelation_scatter_graph(asset_prediction / "autoencoder_scatter_graph.png", raw_filtered, list_of_dates, 0.05)
                    
                    print(f"The files' prediction has been made. Please check {asset_prediction} for the new prediction.\n")
                    
                    
                #check if the user wants to make another prediction using the code below
                user_wants_option = self.yes_or_no_selection("Would you like to make another prediction?([y]es or [n]o).\n")
                print("="*60)
                print("\n") 
                    
            else:
                print("There are no files in the CSV_Files_unseen_dataset folder. Please enter a file in there that you want to predict before starting the model.\n")
                print("="*60)
                print("\n")
                break
    
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
    
    #works don't touch yet
    def machine_learning_pipeline(self,model_created):
        #we need to check if the files are valid again in case of changes
        self.loader.reset_file_validation()
        self.loader.load_and_validate_dataset()#check if the datasets are valid in the unverified folder
        
        self.model_training_ui(model_created)
        
        self.prediction_ui(model_created)
        
        #insert the save model function
        user_wants_option = self.yes_or_no_selection("Would you like to save the model that you have trained to be used again? [y]es or [n]o")
        if user_wants_option:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_directory = Path('saved_models') / f"{model_created.model_name}_{timestamp}"
            model_created.save(model_directory)
            
    def loop_for_ui(self):
        self.copyright_disclaimer()
        model_selected = False
        while(not model_selected):
            attempt_answer = input("Please select one of the following options      [0] Extract stock market data    [1] Isolation Forest  [2] Autoencoder  [3] Load model\n")
            if attempt_answer == '0':
                self.retrieve_stock_market_data()
                
            elif attempt_answer == '1':
                model_selected = True
                self.machine_learning_pipeline(IsolationForestManager(0.01))
                
            elif attempt_answer == '2':
                model_selected = True
                self.machine_learning_pipeline(AutoencoderModelManager())
                
            elif attempt_answer == "3":
                pass #work on loading last
            
            else:
                print("[0], [1], [2], [3] are the only valid inputs please try again).\n")
            print("="*60)
            print("\n") 
        
if __name__ == "__main__":
    ui =  MainControllerUI()
    ui.loop_for_ui()