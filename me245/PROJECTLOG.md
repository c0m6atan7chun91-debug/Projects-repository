# Project Log

Please regularly update this file to record your project progress. You should be updating the project log _at least_ once a fortnight.

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
- Functioning CSV file collector - need to fix file management

---

## Week 4 - 07/02/26

### Memory Management
Implemented a data validation layer designed to handle large files that exceed local RAM capacity. Shifted from full-array loading to a Generator-based memory retrieval system to prevent Out of Memory (OOM) crashes during model training.

### Analytical Constraints
Reached a strategic conclusion regarding missing data: for the purpose of this anomaly detection model, I will assume temporal continuity (Forward Fill). Research suggested that excessive interpolation in a time-series model can lead to "over-smoothing," which would reduce the model's sensitivity to the very anomalies it is designed to find.

### Commit
- Finished data collection and started data verification
---

## Week 5 - 14/02/26

### Temporal Feature Engineering
Researched methods to extract cyclical features from DateTime objects:
- Seasonality  
- Month-of-year  
- Day-of-week  

This is vital because market volatility is often seasonal; for example, energy or commodity shares (like the ice cream analogy) fluctuate based on predictable environmental cycles. Isolating these "normal" seasonal shifts is necessary to prevent them from being flagged as "anomalies."

### Reading
link for reading: https://medium.com/data-science/make-your-machine-learning-model-work-better-with-datetime-features-eb21de397fe8

### Validation Strategy
Researched specific splits for Time-Series data. Unlike standard shuffle splits, temporal data requires:
- Rolling Window split  

This prevents Data Leakage (where the model accidentally "sees" the future during training).

### Reading
link: https://arxiv.org/pdf/1811.12808

<mark>I have been keeping a log!!! Just forgotten to put it into the correct directory so it has now been commited. If you want verification my professor has seen this in the interview before I had made the commit</mark>

