from abc import ABC, abstractmethod

class MachineLearningModel(ABC):
    #ABC = Abstract Base Classs: it only defines what every model must provide
    def __init__(self):
        self.model_name = "Unknown model"
      
    @abstractmethod
    def training_loop(self):
        """Train the model on a list of preprocessed dataframes, one per training file."""
    
    @abstractmethod
    def prediction(self):
        """return anomaly results for one preprocessed unseen dataset."""
    
    @abstractmethod
    def load(self,path):
        """Load a saved model from a file and enter the saved parameters into the model"""
    
    @abstractmethod
    def save(self,path):
        """Write the trained model to a file."""
