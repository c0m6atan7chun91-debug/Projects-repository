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

### Started validation for files
You need to check if the file meets the required format, that is acceptable for the extraction (pandas). So that the file is executable and can be in a correct format for extracting features from raw data. (Appears in the "inital commit").


---

## Week 2 - 24/01/26

### Prototype Development
Created the initial iteration of the data collection script. Encountered synchronization issues when attempting to request multiple tickers simultaneously across long temporal windows.

### Legal & Compliance
Contacted copyright support on the documentation to confirm disclosure requirements for academic redistribution of Yahoo’s data. This is so that I don't have to deal with legal action for improper copyright disclosure (this limited my inital progress with starting the project).

### Technical Challenge
Identified a **"429 Too Many Requests"** error during stress testing; refined the request logic to include a grace period to reduce the time out periods for requesting data. If you request too much they will think that a bot is trying to get data from the server. So I need to make sure there are rules within the algorithm to minimize the time out of requesting data for the user.

### Analytical Constraints
Reached a strategic conclusion regarding missing data: for the purpose of this anomaly detection model. I will assume under the implication that there is no acceptable scenario to fill in missing data in this case of the project. Research suggested that excessive interpolation in a time-series model can lead to "over-smoothing," which would reduce the model's sensitivity to the very anomalies it is designed to find.

### Commit - commited later
initial progress


---

## Week 3 - 31/01/26

### Integration Completion
Successfully finalized the `yfinance` module, allowing for seamless retrieval of OHLCV (Open, High, Low, Close, Volume) data for any asset class required for the model. I need to check for frequent updates for the extension to be able to retreive data. This is due to the fact that they can update communication protocols through the extension.

### Data Imputation Research
Investigated data validation and cleaning techniques such as:
- Linear Interpolation  
- Forward/Backward propagation 
- Moving Averages  

I eventually reached the conclusion that my software must have the limitation that it has to assume the data is complete and must remove rows that have any missing data. As rows that have missing data are impossible to accurately predict and lead to the ML models being skewed with their predictions as the data imputation methods suggested would create a flatline from the last known instance (forward/backward propergation). Another situation is a linear interpolation which would create a unnatural correlation within data by having a consistant upward/ downward intervals until it reaches the next known interval. This lead to the assertion that it is impossible to accurately predict missing data without the cost of model performance. There is also a second software assetion from this which was that all data that the user has in the training folder is already completed and missing values are considered corrupted data.

### Industry Consultation
Asked a professional Actuary to discuss industry-standard assumptions regarding gaps in large datasets. This provided insight into how "noise" is distinguished from actual missing data points in trading environments.

### Commit
- Finished data collection and started data verification

---

## Week 4 - 07/02/26

### Memory Management
Implemented a data validation layer designed to handle large files that exceed local RAM capacity. Shifted from full-array loading to a memory retrieval system to prevent Out of Memory (OOM) crashes during model training. As to avoid future memory issues as this project will be working on RAM for the classical model. So if the user enters masses amounts of data it will be assumed that they have they meet the hardware requirements to do so. I plan to add a instruction function to explain to the user how the software works and what they should do to ensure that their model stays accurate. This is to improve user experience and so that they handle with a more abstract point of view of ML and only need to understand what is needed to ensure accuracy for it from their input datasets.

### File verification
The File validation that will be used throughout the start of the project has been completed. It ensures that it keeps the  OHLCV (Open, High, Low, Close, Volume), it also makes sure the dataset is ordered based on date. This is to ensure that the dataset when entered into the software correctly formats the dataset for extraction as the data is temporal which means that each data point is not independant of one another and is not randomly distributed as they all relate to one another based on when the event occured. This needs to be accurately represented in the data extraction to be able to pick up signals within financial market data. This is so that I don't have the problem later on to tell if it can actually detect anomalies or not within data. If I don't spend large ampunts of time making sure the data extraction works properly it will leads to a more complex problem of actually knowing if I have implemented the data extraction or the isolation forest correctly later on. This would be a nightmare scenario for the project and lead to a lot of reworks and delays later on. I decided to focus more on understanding earlier on rather than later to ensure that everything works properly at each stage rather than creating a pipeline with major issues with logic through out its development. This will be consistent throughout the projects development cycle.

### Validation Strategy
Researched specific splits for Time-Series data. Unlike standard shuffle splits, temporal data requires for deep-learning and data extraction:
- Rolling Window split  

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


<mark>I have been keeping a log!!! Just forgotten to put it into the correct directory so it has now been commited. If you want verification my professor has seen this in the interview before I had made the commit this includes Dr Crole. It is also much shorter in previous weeks as it was building the research foundation required to efficiently complete this project.</mark>

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

I had also started working on the basic visualization of it through using a histogram and a another diagram using matplotlib in order to represent the results. Next week will be implementing SHAP and LIME to further represent the model. I will also start work on the deep learning model, this should be done in two weeks. After that fact I will need to collect data to represent my results and show it can infact detect it and restructure the UI class for a functional terminal interface.

## Week 8 - 08/03/2026

In this week I have managed to fix the interpretation pathing for my interpretation class of my project. This is so that the user can have dedicate folder outcomes named after the users file that they wanted to have predictions for. This gives the user a better method of being able to find and handle the software.

I have had a discussion with my personal supervisor on the diagrams that I had developed and what they mean to the user (such as the histogram and scatter graph). The histogram is to make sure that the software is performing at the 1% significance level of finding outliers (anomalies) within datasets. The scatter graph gives you a visual representation and shows that there are patterns within the stock market data anomalies. This pattern shows that there is correlation within the features that I have used when finding outliers between them. This is positive as it shows I have sucessfully extracted data and reduced the noise from the raw dataset. This is something that I can and will talk about in my dissertation.

I have also done research this week to see how can potentially explore how to find correlating patterns in stock markets using machine learning. My previous background research that I did in weeks 1-4. Point towards using deep-learning methods of ML rather than classical methods of ML. As the deep-learning partitions the dataset into fragements to be processed before hand which enables it to find slight differences within the dataset as it is comparing points from just a segment of data rather than the entire dataset which is what classical machine learning methods use. This lead to me looking into different methods to solve the problem as I don't know what is the best solution from my limited knowledge.

The 2 types of deep-learing that I had considered was the self-attention mechanism from the transformers. As it allows it to weigh each other point against each other and see if there is a positive correlation between them (through using dot product) and it also allows it to understand how each point is relative to one another temporally.However, the problem is that it detects patterns even well known ones to the user. So the user would have to manually decipher whether or not the correlation is an anomaly or not.

 The second type of deep learning I had considered was autoencoders, as they are well known for finding anomalies within data. This is executed through training a NN to be able to reconstruct the data accurately. This is done by using a loss function to be able to tell the difference between the output and the input for the autoencoder. If the difference between the two is significant then it will be flagged as an anomaly. This is why you would ideally want to train deep learning on significantly more amount of data than classical ML. As otherwise without significant amounts of training data it will not understand what is considered normal behaviour from the financial market.

 I also looked into SHAP and LIME documentation. This showed that I didn't really need LIME anymore. SHAP can do the exact same representation as LIME but with better accuracy. This is important as we want to know accurately how much each feature helps to find anomalies within the dataset. As we are already approximating what is an anomaly to the user.

 At the end of the week I had also constructed a plan to create a dissertation from parts of my interem report. I also decided what changes that would be needed from the interem report for it to be suitably applied in the dissertation report. It will be structured as introduction, Survey of literature, Design, Implementation, Testing, and critical appraisal.

## Week 9 - 15/03/2026

This week I reviewed previous project write ups that I have done for other projects to see how I could potentially improve my structure of the dissertation so it can flow well when read and clearly communicate why it was implemented as such. I also created a short list of things I could currently criticize on the project. Such as how much lack of understanding in unsupervised machine learning lead to changes within my project timeline and how this has affected my implementation. I am also going to talk about how it has changed my objectives slightly over time due to my lack of understanding at the time. 

I am also creating a new timeline for myself. This one aims for me to be completed by April the 19th. Including the write up. After that I will be spending the rest of my time creating a presentation for the interview and explain what I had going on in my project and be able to present it to the supervisor and principal marker.

I have also made plans to be able to catch up on the writing on my dissertation for the next few weeks. My aim is to remake parts of the introduction and to further develop my survey of literature. I also plan the week after to finish the survey of literature section and start to further develop my design section. Once I am writing the design section my plan is to code up the autoencoder class and the class that manages the autoencoder and create interpretations of it. (The testing of this development will be included in the testing section of my report and will align with the requirements).

I had to take a break this week like many students; I was experiencing burnout as there isn't much space for breaks within this stage of the degree. So it was a suitable idea so I am able to continue working into April. If you think I could have started in December I couldn't as I was ill for most of the time off we had then.

## Week 10 -  22/03/2026
I had a two modules: Analysis of Algorithms and Cybersecurity group course work given at the same time. As a result of this tight window I had to dedicate all my efforts into these CW. Both of their due dates were around the the 26th and the 30th respectively with 2 weeks to do them.

## Week 11 - 29/03/2026
I had a two modules: Analysis of Algorithms and Cybersecurity group course work given at the same time. As a result of this tight window I had to dedicate all my efforts into these CW. Both of their due dates were around the the 26th and the 30th respectively, with 2 weeks to do them.

Though I had some spare time to see how Lags worked into autocorrelation patterns for my write up. I had researched Lags in Week 5 of this project log.

## Week 12 - 05/04/2026
I had to take the week off to rest up after submitting my last piece of group course work on the 30th of March this week as I am burntout from overloaded work. I have also acknowledged arising health issues from doctors.

## Week 13 - 12/04/2025
I have finished the introduction of my dissertation writing, and I have started my dissertation's literature survey.

## Week 14 - 19/04/2025
The literature survey is finished, which is the longest section to complete.


SHAP diagrams
Deep learning pipeline - using MSE and autoencoders
Optional box plot at the end