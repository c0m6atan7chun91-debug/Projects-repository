import shap
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import EngFormatter
    
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
        figure, axis = plt.subplots(figsize=(20,7)) # width then height
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
        plt.figtext(0.5, 0.01,
            "Distribution of Isolation Forest decision-function scores. "
            "Days scoring below 0 are flagged to be likely anomalous — "
            f"{(predictions < 0).sum()} of {len(predictions)} days "
            f"({100 * (predictions < 0).mean():.1f}%). "
            "The contamination parameter (1%) is an assumed outlier fraction "
            "set at training, so the realised flag rate on unseen data varies.",
            ha="center", fontsize=9, style="italic", wrap=True)
        plt.subplots_adjust(bottom=0.18)
        plt.savefig(path_name)#save figure to a file 
        plt.close()
        
        
    def shap_diagrams(self,path_name, isolation_forest_model, data_extracted, scores):
        explainedModel = shap.TreeExplainer(isolation_forest_model) # tree explainer takes the fitted sklearn model and learns how the tree structure splits the data(becuase of its training). 
        list_of_anomaly_points = data_extracted[scores == -1] #we only want the shap for anomalies
        shap_values_model = explainedModel.shap_values(list_of_anomaly_points)# for each row it explains how much each feature contributed to its anomaly score
        shap.summary_plot(shap_values_model, list_of_anomaly_points,plot_type="bar", show=False)
        plt.savefig(path_name)#save figure to a file 
        plt.close()
        

    def anomaly_autocorrelation_scatter_graph(self, path_name, file_to_predict,
                                        anomalous_window_start_dates,
                                        flag_fraction =0.05, window_size = 15):
        dates = pd.to_datetime(file_to_predict['date'])
        columns_not_date = [ column for column in file_to_predict.columns if column != 'date']
        n_flagged = sum(len(g) for g in anomalous_window_start_dates) #Number of flagged dates within the dataset
        n_windows = len(file_to_predict) - window_size #The limit the windows can start at
        
        #figure represents the whole image and axes is an array of individual plots, one per row.
        #len(columns),1: the grid layout, as rows then columns. You get one row per data column (OHLCAV)
        #and a single column, so the plots sit on top of each other. With a 6x1 grid.
        #figsize() the figure size in inches, width then height
        #sharex = true :all the plots share the same axis. Zooming or setting limits apply to all plots. Hides the date labels from all but the bottom one. A shared red band therefore sits at the same position.
        figure, axes = plt.subplots(len(columns_not_date), 1,
                                    figsize=(16, 3 * len(columns_not_date)), sharex=True)
        
        
        #Zip() takes two or more lists and walks through them together. On each step ti hands you one item from each list as  a pair.
        # set_major_formatter() - Sets the function that turns each labelled tick value into text.
        # ENgFormatter - It writes numbers with metric prefixes, like engineering notation e.g. 2,500 = 2.5k and 40,000,000 = 40M
        for axis, column in zip(axes, columns_not_date):
                    axis.scatter(dates, file_to_predict[column], color='steelblue', s=12)
                    for group in anomalous_window_start_dates:
                        start = dates[group[0]]
                        end   = dates[min(group[-1] + window_size, len(dates) - 1)]# chooses the last group flagged or last possible position to flag
                        axis.axvspan(start, end, color='red', alpha=0.2)
                    axis.set_ylabel(column)
                    if column == 'volume':
                        axis.yaxis.set_major_formatter(EngFormatter())
        
        # band start-date annotations on the top panel only
        top = axes[0] #selects the first plot
        for group in anomalous_window_start_dates:
            start = dates[group[0]]
            #top.get_ylim() - returns the current y-axis range of the top plot as a pair (bottom, top)
            top.annotate(str(start.date()), xy=(start, top.get_ylim()[1]),
                            rotation=90, fontsize=7, va='top', ha='right') # last two variables are postions
            
        #Locatior decides where ticks are placed and formatter decides what text each tick shows.
        #mdates.MonthLocator(interval=2): puts a tick on the 1st of every 2nd month.
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))#Sets where the tick marks go on the bottom plot's date axis.
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(axes[-1].get_xticklabels(), rotation=90, fontsize=8)
        axes[-1].set_xlabel("date")
        
        #sets a title for the whole figure: one heading centred above all 6 plots. "Sup" is short for "super", as in a title over the others.
        figure.suptitle(f"{path_name.parent.name}: anomalous {window_size}-day "
                        f"windows shaded (autoencoder)", y=0.995)
        figure.text(0.5, 0.005,
                    f"Shaded: top {flag_fraction:.0%} of {n_windows} windows by "
                    f"reconstruction error ({n_flagged} flagged). Top-panel labels "
                    f"give window start dates.",
                    ha="center", fontsize=9, style="italic", wrap=True)
        figure.savefig(path_name.parent / "all_features.png",
                        bbox_inches="tight", dpi=130)
        plt.close(figure)