# Periodic Signal Forecasting

## 1. Data Choice

Three versions of the same time series were provided: 30-minute, 1-hour, and 5-hour sampling.

The EDA notebook showed that all three datasets are clean, regular, and contain the same main repeating pattern. Autocorrelation and FFT analysis both identified a dominant period of about **274.1 hours (11.42 days)** across all three datasets.

The **5-hour dataset** was selected because it still gives about **55 observations per cycle**, which is enough to represent the repeating pattern. The values at common timestamps also have about **0.998 correlation** with the finer datasets. The within-5-hour variation is relatively small compared with the overall variation of the signal.

Therefore, 5-hour sampling keeps the main trend and periodic structure while reducing unnecessary data volume and computation.

See `notebooks/01_data_exploration.ipynb` for the full EDA and sampling comparison.

---

## 2. Feature and Target Definition

The target is the observed signal value:

**Target:** `value`

The final model uses six features:

- `time_normalized`
- `time_squared`
- `sin_cycle`
- `cos_cycle`
- `time_sin_cycle`
- `time_cos_cycle`

The features were chosen based on the patterns found during EDA.

The signal has a long-term trend, but the trend is not well represented by a simple straight line. The `time_normalized` and `time_squared` features allow the model to learn a curved trend.

The signal also has a strong repeating cycle. Sine and cosine features represent the position of the signal within this cycle:

$$
\theta_t = \frac{2\pi t}{P}
$$

where `P` is the dominant period in observations.

The EDA estimated the period as approximately **54.824 observations** for the 5-hour data.

The interaction features `time × sin_cycle` and `time × cos_cycle` allow the size of the repeating wave to change over time. This was useful because the amplitude of the observed cycles is not completely constant.

The final model can therefore be viewed as learning:

> **a curved long-term trend + a repeating cycle + a cycle whose amplitude changes over time.**

---

## 3. Model Experimentation

A chronological validation set containing the final six months of historical data was used to compare different approaches.

| Model | MAE | RMSE | Decision / Reason |
|---|---:|---:|---|
| Naive | 2.8456 | 3.1884 | Baseline; cannot capture the repeating pattern |
| Seasonal Naive | 1.8029 | 2.1570 | Better, but simply repeats one cycle and cannot adapt to the changing trend |
| Harmonic regression + trend | 1.5124 | 1.8422 | Captures the main cycle, but a simple linear trend is not enough |
| Harmonic regression + 2nd harmonic | 1.5121 | 1.8422 | Almost no improvement; extra harmonic not useful at this stage |
| Harmonic regression + changing amplitude | 1.4045 | 1.7447 | Better; shows that the seasonal amplitude changes over time |
| Lag-based Linear Regression | 3.5723 | 3.8115 | Recursive forecasting causes error accumulation over the long horizon |
| Harmonic regression + nonlinear trend | **0.8703** | **1.0974** | **Selected final model** |
| Harmonic regression + nonlinear amplitude | 0.8155 | 1.0573 | Better validation score, but produced an overly aggressive six-month extrapolation |
| Gradient Boosting | 1.7832 | 2.1456 | Worse than the engineered linear model for this smooth signal |
| Random Forest | 1.8491 | 2.2138 | Worse than the engineered linear model for this smooth signal |
| + 2nd/3rd Fourier harmonics | 0.8705 | 1.0973 | Negligible change; confirms that the main frequency is enough |

**Note:** Harmonic-order experiments were repeated at two stages — once with the basic trend model and again after adding the nonlinear trend — to confirm that the single dominant frequency remained sufficient.

### Why did the tree-based models perform worse?

Random Forest and Gradient Boosting are useful models, but this signal has a different structure.

Our features describe the signal using smooth mathematical functions: time, a curved trend, sine/cosine waves, and their interactions.

Linear Regression can directly combine these features to form a smooth curve and continue that learned relationship into the future.

Tree-based models instead divide the feature space into regions and make piecewise predictions. This is less natural for a smooth periodic signal. They are also generally weaker at extrapolating a trend beyond the range of values seen during training.

Therefore, the weaker tree-model performance is specific to this problem. It does not mean that tree models are generally worse than linear models.

### Why was the more complex nonlinear-amplitude model rejected?

The model with `time² × sin_cycle` and `time² × cos_cycle` had the best validation error among the tested models.

However, its six-month forecast became too aggressive when the polynomial amplitude terms were extrapolated beyond the training period.

Although this model had a better validation score (MAE 0.8155 vs. 0.8703), the forecast also needs to remain stable and reasonable over the full six-month horizon.

The simpler six-feature model was therefore selected because it provided a better balance between validation performance, interpretability, and long-term forecast stability.

## 4. Validation Strategy

A random train/test split was not used because this is a time-series forecasting problem.

The data was split chronologically:

- **Training:** January 2016 to June 2017
- **Validation:** approximately July 2017 to December 2017

The model only saw earlier observations when learning and was then tested on later observations.

This is closer to the real forecasting situation and avoids using future information to predict the past.

---

## 5. Prediction Strategy

The final model uses **direct multi-step forecasting**.

The model is not autoregressive. It does not need the previous predicted value to make the next prediction.

Instead, future timestamps are created first, and the six features are calculated for every future timestamp. The fitted model then predicts all future points directly in one batch.

For the final forecast:

- Historical observations: **3,509**
- Sampling interval: **5 hours**
- Forecast horizon: **at least 6 calendar months**
- Future prediction points: **869**

This approach is appropriate because the final model is a function of time and periodic phase.

It also avoids the error accumulation seen in the recursive lag-based experiment, where each prediction was used as an input for the next prediction.

---

## 6. Normalization

The time index was normalized using the maximum historical time index as the scale:

$$
t_{normalized} = \frac{t}{t_{max}}
$$

For the full dataset, the historical time index runs from `0` to `3508`, so the final historical value is scaled to `1`.

We then created the nonlinear trend feature:

$$
t_{squared} = t_{normalized}^2
$$

The same historical `t_max` value (`3508`) was used when creating future features. This keeps the training and forecast features on the same scale.

The normalization was mainly used to keep the polynomial and interaction features numerically well-scaled during regression.

---

## 7. Uncertainty

No formal prediction interval was included in the final output.

The assignment mainly requires the six-month point forecast, so the final implementation focuses on producing a clear and stable forecast rather than adding an uncertainty method that was not fully validated for this signal.

A future improvement would be to estimate forecast uncertainty using a suitable bootstrap or residual-based approach.

---

## 8. Limitations

The final model assumes that the main trend and periodic structure observed in the historical data will continue into the forecast period.

This is reasonable for this dataset because the repeating pattern is strong and consistent across the available history, but it cannot guarantee the actual future behavior.

The further we forecast into the future, the more uncertainty there is around the trend and amplitude. The model also uses only the historical signal itself and does not include external variables that could explain future changes.

The model should therefore be viewed as a structured extrapolation of the observed signal, not as a guarantee of the exact future values.

---

## 9. Project Structure

```text
signal-forecasting-assignment/
│
├── data/
│   └── raw/
│       ├── timeseries_30m.csv
│       ├── timeseries_1h.csv
│       └── timeseries_5h.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_experiments.ipynb
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── evaluation.py
│   └── visualization.py
│
├── plots/
│   └── final_forecast.png
│
├── main.py
├── requirements.txt
└── README.md
```

The notebooks contain the analysis and model experiments. The `src/` files contain the final reusable pipeline. `main.py` runs the final forecasting process and saves the required visualization.

---

## 10. How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the final forecasting pipeline:

```bash
python main.py
```

The script trains the final model on all available historical data, creates a forecast covering at least six calendar months, and saves the final plot to:

```text
plots/final_forecast.png
```
