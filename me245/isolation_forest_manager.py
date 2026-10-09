from sklearn.ensemble import IsolationForest
from machine_learning_model import MachineLearningModel
import joblib



class IsolationForestManager(MachineLearningModel):
    def __init__(self, contamination_input):
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
        super().__init__()
        self.model_name  = "Isolation Forest"
        self.isolation_forest = IsolationForest(n_estimators = 200, contamination=contamination_input, n_jobs=-1)
    
    def training_loop(self,train):
        #Each time .fit() is called completely overwrites it previous training each time it is called so it needs to train all in one go
        self.isolation_forest.fit(train) #O(nlogn) time complexity for training
    
    def prediction(self,unseen_data):
        #For each observation, tells whether or not (+1 or -1) it should be considered as an inlier according to the fitted model.
        scores = self.isolation_forest.predict(unseen_data)
        #The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.
        prediction = self.isolation_forest.decision_function(unseen_data)#the list of
        return scores, prediction
    
    def save(self, path):
        path.parent.mkdir(parents=True,exist_ok=True)
        path = path.with_suffix(".joblib")
        joblib.dump(self.isolation_forest, path)
    
    def load(self, path):
        path.parent.mkdir(parents=True,exist_ok=True)
        self.isolation_forest = joblib.load(path)