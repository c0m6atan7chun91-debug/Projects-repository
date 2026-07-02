# A.D.F.M.D. — Anomaly Detection in Financial Market Data

A terminal-driven Python application that detects anomalies in financial market time-series data using two unsupervised machine learning models: an **Isolation Forest** (classical ML) and a **deep learning Autoencoder** (PyTorch).

---

## What it does

The system ingests OHLCV (Open, High, Low, Close, Volume) financial datasets, engineers a set of temporal and statistical features from the raw price data, and then trains one of two anomaly detection models across the entire training corpus. Once trained, it can flag unusual trading days or periods in any new (unseen) dataset and produce interpretable visual outputs.

Anomalies surfaced by the models can indicate events such as flash crashes, unusual volume spikes, potential insider trading build-up, or other market irregularities that deviate from learned normal behaviour.

---

## How it works

### 1. Data collection (`data_loader.py`)

- **Automatic download**: Enter a ticker symbol (e.g. `AAPL`, `^GSPC`) and a date range. The system fetches OHLCV data via the Yahoo Finance API (`yfinance`) with rate-limiting safeguards (2-second delay between requests, graceful retry on 429 errors). Downloaded files are saved as `<TICKER>_<start>_<end>.csv` (e.g. `AAPL_2015-01-01_2021-01-01.csv`), so the asset and date range covered are immediately identifiable from the filename alone.
- **Manual CSV**: Bring your own CSV files — place them in the unverified training folder and the system validates and promotes them automatically. Custom files should follow the same `<name>_<start>_<end>.csv` naming convention so that the date range of expected outcomes remains clear from the filename.
- **Validation**: Every file must contain the required columns (`date`, `open`, `high`, `low`, `close`, `adjclose`, `volume`), have at least 1 000 trading days, contain no missing values, and be ordered chronologically. Files that fail are left in the unverified folder.

### 2. Feature engineering (`preprocessor.py`)

Raw OHLCV values are transformed into 15 normalised features per trading day:

| Feature group | Description |
|---|---|
| Log returns | Day-over-day log change for `adjclose`, `open`, `high`, `low`, `close`, and `volume` — scale-invariant measure of price movement |
| Rolling window divergence | Difference between short, medium, and long rolling means for `volume` and `adjclose` — detects unusual momentum shifts |
| Intraday ratios | `(open − close) / open` and `(high − low) / open` — captures intraday volatility structure |
| Temporal features | `year`, `month`, and `season` — lets the model learn expected seasonal patterns and avoid flagging predictable cycles as anomalies |

All features are z-score normalised before training so that datasets from different markets can be concatenated and compared on the same scale.

### 3. Model training

**Option 1 — Isolation Forest** (`classical_model_manager.py`)

- All validated training files are feature-extracted and concatenated into a single DataFrame.
- An `IsolationForest` (200 trees, 1% contamination, parallelised) is fitted on the combined dataset.
- The model learns a global notion of "normal" market behaviour across all supplied assets.

**Option 2 — Autoencoder** (`autoencoder.py`, `autoencoder_model_manager.py`)

- Each training file is processed into overlapping 60-day sliding windows (sequences), giving the model a local temporal context equivalent to one financial quarter.
- Each 60-day window (60 days × 15 features = 900 values) is fed through a fully-connected encoder–decoder network trained with MSE loss to reconstruct its own input.
- Architecture: `900 → 90 → 45` (encoder) then `45 → 90 → 900` (decoder), all with Tanh activations.
- Training runs for 20 epochs per file in mini-batches of 32 windows, using the Adam optimiser (lr = 0.0001).
- At inference time, sequences whose reconstruction error exceeds the mean + 2.33 standard deviations (99% one-tailed threshold) are flagged as anomalous.

### 4. Prediction & interpretation (`interpretation.py`)

After training, you select any `.csv` file from the unseen dataset folder and the system produces:

**For Isolation Forest:**
- `scatter.png` — anomaly score of every trading day plotted over time, with anomalous points highlighted in red
- `histogram.png` — distribution of decision-function scores for normal vs. anomalous days, confirming the 1% significance level
- `shap.png` — SHAP bar chart showing which engineered features most contributed to each anomaly flag

**For Autoencoder:**
- Per-column scatter plots with anomalous 60-day windows shaded in red, so you can inspect which market variable drove the reconstruction failure

All outputs are saved to `me245/File_of_outcomes/<model_type>/<asset_name>/`.

---

## Directory structure

```
me245/
├── main.py                          # Entry point and terminal UI
├── data_loader.py                   # Yahoo Finance download + CSV validation
├── preprocessor.py                  # Feature engineering and sequence creation
├── classical_model_manager.py       # Isolation Forest wrapper
├── autoencoder.py                   # PyTorch autoencoder network definition
├── autoencoder_model_manager.py     # Training loop and prediction logic
├── interpretation.py                # SHAP, histogram, and scatter visualisations
│
├── CSV_Files_training_unverified/   # Drop raw training CSVs here
├── CSV_Files_training_verified/     # Validated training files (managed automatically)
├── CSV_Files_unseen_dataset/        # CSVs to run predictions on
└── File_of_outcomes/                # Prediction outputs, organised by model and asset
```

---

## Output diagrams

All diagrams are saved as `.png` files under `me245/File_of_outcomes/<model_type>/<asset_name>/`. The output folder name is taken from the stem of the unseen CSV filename, so the asset and date range of the prediction are encoded directly in the folder path — for example, predictions for `AAPL_2015-01-01_2021-01-01.csv` are saved under `File_of_outcomes/Isolation Forest/AAPL_2015-01-01_2021-01-01/`.

Outputs are intentionally limited to `.png` charts rather than supplementary CSV exports. The software is designed for users with a financial or quantitative background who can read and interpret the visualisations directly. The charts convey distributional shape, temporal clustering, and feature attribution in a way that is faster to assess than a flat list of flagged rows, and they are the standard medium for presenting anomaly detection results in a quantitative finance context.

### Isolation Forest

#### `scatter.png` — Anomaly score scatter plot

The x-axis is the date index of each trading day in the unseen dataset. The y-axis is the Isolation Forest **decision function score** — a continuous value where more negative means more anomalous. A horizontal black line sits at y = 0, which is the model's boundary between inlier and outlier territory.

- Normal trading days are plotted in **steel blue**
- Days classified as anomalies (decision score below zero) are plotted in **red**
- The x-axis ticks show only the dates of flagged anomaly days, rotated 90° so they remain readable

**What to look for:** clusters of red points in narrow time windows often correspond to real market events (crashes, sudden volume surges, earnings shocks). A scatter of isolated red points across many dates may indicate the contamination rate needs tuning, or that the training corpus does not cover that market's normal behaviour well.

---

#### `histogram.png` — Score distribution histogram

A frequency histogram that overlays two distributions:

- **Sky blue** bars: decision function scores for all days classified as **normal**
- **Salmon** bars: decision function scores for all days classified as **anomalies**
- A vertical black line at x = 0 marks the decision boundary

**What to look for:** the two populations should be clearly separated by the black line, with the anomaly bars pushed into negative score territory and the normal bars clustered in positive territory. Well-separated distributions confirm that the 1% contamination rate is producing a meaningful split rather than arbitrary labelling. Overlap between the two groups is a sign that the model is uncertain around the boundary, and that some flagged events may not be genuine anomalies.

---

#### `shap.png` — SHAP feature importance bar chart

Generated using `shap.TreeExplainer` applied to the fitted Isolation Forest. It only considers the rows that were **classified as anomalies** (score = −1) and shows how much each of the 15 engineered features contributed to pushing those data points toward being labelled as outliers.

- Bars are sorted by mean absolute SHAP value from largest to smallest
- A longer bar means that feature had greater influence on the anomaly classification

**What to look for:** features with the largest bars are the primary drivers of the anomalies found. For example, if `log_volume_change` or `short_mid_diff_volume` dominates, the flagged events are largely volume-driven. If `log_adjclose_change` or `difference_high_low` leads, the anomalies are price-movement-driven. This tells you the *nature* of the market irregularity, not just its location.

---

### Autoencoder

#### `<column_name>.png` — Per-feature autocorrelation scatter plot (one file per OHLCV column)

One diagram is produced for every raw column in the unseen CSV (open, high, low, close, adjclose, volume). Each plot shows:

- The raw daily value of that column on the y-axis against date on the x-axis, plotted as **steel blue** dots
- **Red shaded bands** (semi-transparent) overlaid on every 60-day window whose reconstruction error exceeded the 99% anomaly threshold — consecutive anomalous windows are merged into a single wider band
- The x-axis ticks mark the start dates of every anomalous sequence, rotated 90° for readability
- The x-axis is cropped to the span between the first and last anomalous window, removing quiet periods at the edges to focus the view

**What to look for:** a red band that lines up with a visible spike, dip, or sudden regime change in the plotted column confirms that the autoencoder has identified a real structural break in the data. A red band over a visually calm section of a chart may indicate an anomaly in a *different* feature within that same 60-day window — because the autoencoder evaluates all 15 features simultaneously, the anomaly might be driven by, say, a volume surge even when the close price looks ordinary. Comparing the shaded windows across all column plots for the same asset helps identify which variable was the primary driver of each flagged period. The dates at the bottom are the start of said windows.

---

## Why SHAP is not used for the Autoencoder

SHAP's `TreeExplainer` works by inspecting the internal split structure of tree-based models (such as the Isolation Forest's ensemble of decision trees) to compute exact, mathematically grounded attributions for each feature. It can do this efficiently because each prediction in a tree model traces a deterministic path through a fixed set of binary splits.

The Autoencoder is a neural network. Its "decision" — whether a 60-day window is anomalous — is not based on a single forward prediction but on the **magnitude of the error** between the network's input and its reconstructed output after passing through hundreds of continuous, non-linear weight operations. There is no tree structure for `TreeExplainer` to inspect.

SHAP does provide gradient-based explainers for neural networks (`DeepExplainer`, `GradientExplainer`), but these explain why the network produced a particular *output value* — in this case, the reconstructed sequence. They do not directly explain why the **reconstruction error** was high for a given window, which is the actual anomaly signal. Applying them here would explain the reconstruction, not the anomaly, making the output misleading rather than informative.

For this reason, the Autoencoder's interpretability is instead handled visually: the per-column scatter plots with shaded anomalous windows allow the user to inspect the raw market data at the time of each flagged period and draw their own conclusion about which variable drove the reconstruction failure.

---

## Hardware requirements

The project is pinned to **PyTorch 2.5.1 with CUDA 12.1**, meaning it expects an NVIDIA GPU with CUDA 12.1 support for autoencoder training. On a CUDA-capable machine the training loop runs on the GPU automatically.

If no compatible GPU is available the autoencoder will fall back to CPU. This will work correctly but training will be significantly slower, as the model trains sequentially on each file in the corpus (20 epochs per file) with no parallelism at the file level.

The Isolation Forest is CPU-bound regardless and uses all available cores (`n_jobs=-1`), so hardware has minimal impact on its training time.

There are no strict minimum RAM requirements enforced by the code, but loading and concatenating large numbers of high-frequency CSV files during Isolation Forest training is done entirely in memory. A machine with at least 8 GB RAM is advisable if the training corpus is large.

---

## Development status

This project is currently a **work in progress**. The two model pipelines (Isolation Forest and Autoencoder) are deliberately kept as separate, independent classes rather than being unified under a shared base class or interface.

Introducing polymorphism — for example, a common `Model` base class with shared `train` and `predict` methods — would be the natural next step, but it has been deferred until **model persistence (save and load)** is implemented. The two models have meaningfully different save requirements: the Isolation Forest would be serialised with `joblib`, while the Autoencoder requires `torch.save` and `torch.load`. Abstracting these behind a shared interface before the save/load behaviour is finalised would mean designing the abstraction around incomplete behaviour, risking a structural rework once persistence is added.

The planned sequence is:
1. Implement save and load for both models
2. Define a shared interface once the full contract (train, predict, save, load) is known
3. Refactor both managers to implement it

Until then, duplication between the two pipelines is intentional and preferable to a premature abstraction.

---

## Running the application

Run from the **project root directory** (the folder above `me245/`), not from inside `me245/`:

```bash
python me245/main.py
```

The terminal menu presents three options:

```
[0] Extract stock market data    — download data from Yahoo Finance
[1] Isolation Forest             — train and predict with the classical model
[2] Autoencoder                  — train and predict with the deep learning model
```

**Minimum requirements for training:** at least 10 valid CSV files in `CSV_Files_training_unverified/`, each with 1 000+ rows of complete OHLCV data.

---

## Dependencies

Key libraries (see `requirements.txt` for the full pinned environment):

| Library | Purpose |
|---|---|
| `yfinance` | Yahoo Finance data download |
| `pandas`, `numpy` | Data manipulation and feature engineering |
| `scikit-learn` | Isolation Forest |
| `torch` (PyTorch 2.5, CUDA 12.1) | Autoencoder neural network |
| `shap` | Model interpretability |
| `matplotlib` | Visualisation |

---

## Data assumptions

- Missing values are treated as corrupted data and the containing row is dropped. No interpolation is performed, as filling gaps artificially smooths the very signals the model is trained to detect.
- All training CSVs are assumed to be complete (as produced by `yfinance`).
- The model is **generalised** — it learns normal behaviour across a diverse corpus of assets rather than per-ticker. This improves robustness but means it flags anomalies relative to broad market norms, not asset-specific baselines.

---

## Legal notice

This program uses the `yfinance` library to access data via the Yahoo Finance API. `yfinance` is not affiliated with Yahoo. Use of the data is subject to Yahoo's Terms of Service: https://policies.yahoo.com/us/en/yahoo/terms/index.htm. See also the yFinance project page: https://pypi.org/project/yfinance/.
