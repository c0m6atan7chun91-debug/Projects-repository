import shap
import matplotlib.pyplot as plt
    
class Interpretation:
    def interpretation_collection(self, scores, predictions,data_extracted, path_name, isolation_forest_model):
        #with_name() allows me to change the name of the stem path
        self.shap_diagrams(path_name / "shap.png", isolation_forest_model, data_extracted, scores)
        self.anomaly_score_histogram( scores, predictions, path_name / "histogram.png")
        self.scatter_graph_representation( scores, predictions,data_extracted, path_name / "scatter.png")
        
    def scatter_graph_representation(self, scores, predictions,data_extracted, path_name):
       
        # Add scores and labels to dataframe
        data_extracted['anomaly_score'] = predictions
        data_extracted['is_anomaly'] = scores
        filtered_anomaly = data_extracted[data_extracted['is_anomaly'] == -1] #this line also keeps the corresponding rows index after the filter to make sure it matches with the dataextracted df
        #subplots() returns two thing figure-the overall container and axes - the actual plot area where you draw things
        figure, axis = plt.subplots(figsize=(15,7)) # width then height
        #scatter call for normal points
        axis.scatter(data_extracted.index, data_extracted['anomaly_score'],color = 'steelblue',alpha = 1)#x-axis then y-axis and alpha is the transparency
        #scatter call for anomaly points
        axis.scatter(filtered_anomaly.index, filtered_anomaly['anomaly_score'],color = 'red',alpha = 1)#normal is +1 and an anomaly is -1
        axis.axhline(y = 0,color = 'black', linewidth=1) # draws a horizontal line across the entire plot at a given y value. y being where the line sits(threshold),color of line, linewidth,linestyle,label - text shown for legend
        axis.set_xlabel("anomaly date")#xaxis label, font size, font weight, font colour
        axis.set_ylabel("anomaly score")
        plt.xticks(filtered_anomaly.index,rotation = 90, fontsize= 7)
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
        plt.close()
        
        
    def shap_diagrams(self,path_name, isolation_forest_model, data_extracted, scores):
        explainedModel = shap.TreeExplainer(isolation_forest_model) # tree explainer takes the fitted sklearn model and learns how the tree structure splits the data(becuase of its training). 
        list_of_anomaly_points = data_extracted[scores == -1] #we only want the shap for anomalies
        shap_values_model = explainedModel.shap_values(list_of_anomaly_points)# for each row it explains how much each feature contributed to its anomaly score
        shap.summary_plot(shap_values_model, list_of_anomaly_points,plot_type="bar", show=False)
        plt.savefig(path_name)#save figure to a file 
        plt.close()
        
    
    def anomaly_autocorrelation_scatter_graph(self, path_name, file_to_predict, anomalous_window_start_dates):
        for column in file_to_predict.columns:
            if column == 'date':
                continue
            #subplots() returns two thing figure-the overall container and axes - the actual plot area where you draw things
            figure, axis = plt.subplots(figsize=(30,5)) # width then height
            #scatter call for normal points
            axis.scatter(file_to_predict['date'], file_to_predict[column],color = 'steelblue',alpha = 1)#x-axis then y-axis and alpha is the transparency
            #scatter call for adjclose points to plot
            for group in anomalous_window_start_dates:
                #required to highlight sections on the graph with anomalous autocorrelation
                plt.axvspan(group[0], group[-1] + 60, color='red', alpha=0.2)
            dates = [file_to_predict['date'][index] for group in anomalous_window_start_dates for index in group]#this line plots the indexs that are anomolous onto the x-axis
            plt.xticks(dates, rotation = 90,fontsize=7)# plots all starting points of the sequences (60 days) onto the graph
            axis.axhline(y = 0,color = 'black', linewidth=1) # draws a horizontal line across the entire plot at a given y value. y being where the line sits(threshold),color of line, linewidth,linestyle,label - text shown for legend
            axis.set_xlabel("date")#xaxis label, font size, font weight, font colour
            axis.set_ylabel(f"{column}")
            axis.set_xlim(file_to_predict['date'][anomalous_window_start_dates[0][0]],file_to_predict['date'][anomalous_window_start_dates[-1][-1]+ 60])
            plt.title("Diagram of the varying anomaly scores against index occurance")
            plt.tight_layout()#Prevents lables from being cut off
            plt.savefig(path_name.parent / f"{column}.png")#save figure to a file 
            plt.close()
    