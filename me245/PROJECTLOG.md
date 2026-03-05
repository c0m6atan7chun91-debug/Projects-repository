# Project Log

Please regularly update this file to record your project progress. You should be updating the project log_at least_ once a fortnight.

# Project Log: Anomaly Detection in Financial Markets

---

## Week 1 - Semester 2 - 17/01/26

### API Infrastructure Research
Investigated the `yfinance` extension to facilitate the automated retrieval of high-volume financial datasets. Analyzed the library's capability to handle ticker and Download objects for batch processing.

### Rate-Limiting & Ethics
Researched API communication syntax to mitigate the risk of IP blacklisting from Yahoo Finance’s backend. Implemented `requests_cache` and time-delay headers to ensure the data scraper adheres to professional practices. This was a critical preventative measure; losing access to the backend would cause significant delays in the training phase and project creation.

### Reading
**Fetching Real-Time Gold Prices from Yahoo Finance**  
Utilized this to understand the mapping of raw JSON responses to structured Pandas DataFrames.

link For example: https://medium.com/@hfahmida/fetching-real-time-gold-prices-from-yahoo-finance-and-visualizing-trends-python-944181de1dd2

### Commit - commited later
initial progress

---

## Week 2 - 24/01/26

### Prototype Development
Created the initial iteration of the data collection script. Encountered synchronization issues when attempting to request multiple tickers simultaneously across long temporal windows.

### Legal & Compliance
Contacted copyright support on the documentation to confirm disclosure requirements for academic redistribution of Yahoo’s data.

### Technical Challenge
Identified a **"429 Too Many Requests"** error during stress testing; refined the request logic to include exponential backoff.

### Analytical Constraints
Reached a strategic conclusion regarding missing data: for the purpose of this anomaly detection model. I will assume under the implication that there is no acceptable scenario to fill in missing data in this case of the project. Research suggested that excessive interpolation in a time-series model can lead to "over-smoothing," which would reduce the model's sensitivity to the very anomalies it is designed to find.


---

## Week 3 - 31/01/26

### Integration Completion
Successfully finalized the `yfinance` module, allowing for seamless retrieval of OHLCV (Open, High, Low, Close, Volume) data for any asset class required for the model.

### Data Imputation Research
Investigated data validation and cleaning techniques such as:
- Linear Interpolation  
- Forward/Backward propagation 
- Moving Averages  

### Industry Consultation
Asked a professional Actuary to discuss industry-standard assumptions regarding "stochastic" gaps in large datasets. This provided insight into how "noise" is distinguished from actual missing data points in trading environments.

### Commit
- Finished data collection and started data verification

---

## Week 4 - 07/02/26

### Memory Management
Implemented a data validation layer designed to handle large files that exceed local RAM capacity. Shifted from full-array loading to a memory retrieval system to prevent Out of Memory (OOM) crashes during model training. As to avoid future memory issues as this project will be working on RAM for the classical model.

### File verification
The File validation that will be used throughout the start of the project has been completed. It ensures that it keeps the  OHLCV (Open, High, Low, Close, Volume), it also makes sure the dataset is ordered based on date as to make sure the fact that each data point is not independant of the other

### Validation Strategy
Researched specific splits for Time-Series data. Unlike standard shuffle splits, temporal data requires:
- Rolling Window split  

This prevents Data Leakage (where the model accidentally "sees" the future during training).

### Reading
link: https://arxiv.org/pdf/1811.12808

### Commit
- File loading and verification finished
---

## Week 5 - 14/02/26

### Temporal Feature Engineering
Researched methods to extract cyclical features from DateTime objects:
- Seasonality  
- Month-of-year  
- Day-of-week  

This is vital because market volatility is often seasonal; for example, energy or commodity shares (for example a ice cream company) fluctuate based on predictable environmental cycles. Isolating these "normal" seasonal shifts is necessary to prevent them from being flagged as "anomalies." I had also covered other methods of data extraction such as log returns to measure it.

### Reading
link for reading: https://medium.com/data-science/make-your-machine-learning-model-work-better-with-datetime-features-eb21de397fe8


<mark>I have been keeping a log!!! Just forgotten to put it into the correct directory so it has now been commited. If you want verification my professor has seen this in the interview before I had made the commit this includes Dr Crole</mark>

## Week 6 - 22/02/26

### Data extraction
This week mostly covered the data extraction and how it works in association to the machine learning model and what were common practises to this. So in this case the log returns are better than precentage returns, as it doesn't increase to a large number if significantly bigger. This can skewe a model, if the numbers aren't correctly normalized in relation to the rest of the population, as it can weigh those numbers far more significantly than others. Having massive neumerical skewes in models can cause them to over look other anomalies within the dataset that perhaps have a much smaller and niche significance in relation to that outlier. For example:

*** equation for percentage return(P - stands for data point and t stands for the point of time in relation to said point):**
(Pt - p(t-1))/ p(t-1) example: (100 - 90)/90 = 0.111%(3 D.P.), (80 - 90)/90 = -0.111%(3 D.P.), (10000-90) /90 = 110.111% (3 D.P.)

while log returns for example would have the equation(log -assumed as base 10):
log(Pt/P(t-1))
example of application: Log(10/1) = 1, Log(20000/10) = 3.301 (3 D.P.), log(80/90) = -0.051 (3 D.P.)

Based off of these calculations they clearly demonstrate how much it still shows significant outliers in relation to other data points but without having to use large numbers to quantify the difference. I then normalized the data so that it can further reduce the neumerical scaling difference as that can still be an issue. 

I also used my ideas and asked AI to initially generate its own version to see if there were ways of identifying the anomalies that I hadn't considered. There were it introduced intraday ratio, lower shadow and upper shadow as necessary features for me to understand. I will now need to study these features and gradually implement them.

## Week 7 - 01/03/2026
During this week I tried to see the outcome of a 80/20 split of Isolation Forest, when trained on a single dataset, this was unsuitable for the model to learn what was normal behaviour in the stock market. This was due to the insufficient variety in data given to the model leading it to flag outliers when none existed. In simpler terms it was returning false positives (flagging normal data points as anomalies) and false negatives (missing genuine anomalies) in the dataset, which is a characteristic of a poor model.

This led to me coming to the solution to just data extract all the datasets given, which were no more than 50 (from diverse markets). I would then data extract them individually and store them into a list of dataframes that were extracted. Afterwards I would concatenate the dataframes together so that the model can train them all together. However, this results in a more generalized model, but is required if we want a model to be able to capture general anomalies in all markets rather than a stock specific one.

I had also started working on the basic visualization of it through using a histogram and a another diagram using matlab in order to represent the results. Next week will be implementing SHAP and LIME to further represent the model. I will also start work on the deep learning model, this should be done in two weeks. After that fact I will need to collect data to represent my results and show it can infact detect it and restructure the UI class for a functional terminal interface.

## Week 8 - 01/03/2026
Fixed the path issue in interpretation_collection


SHAP diagrams
Deep learning pipeline - figured out how it works and what model is required
Optional box plot at the end