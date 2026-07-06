# Climate-Aware Crop Yield Forecasting: Baseline Comparison Metrics

To establish the real-world forecasting efficacy of the **MultiModal Transformer + MDN** platform, we evaluated it against several traditional and machine learning baseline models. 

Evaluation was conducted using a 5-fold cross-validation split over historical yields from Burdwan, West Bengal and surrounding study regions.

## Baseline Descriptions

1. **Historical Average (Naive Baseline)**
   - Simply predicts the historical mean yield for the specific region across all past years. It represents a zero-information baseline.
2. **Ridge Regression (Tabular Baseline)**
   - A linear regression model with L2 regularization. It is trained on flattened average satellite index (NDVI), raw season-wide weather aggregates (mean temperature, total precipitation), and static soil attributes.
3. **XGBoost (Tabular Machine Learning)**
   - An optimized gradient-boosting tree regressor. Operates on tabularized temporal means and static features, capturing non-linear static-feature correlations but lacking raw sequence awareness.
4. **Standard LSTM (Weather-only Temporal Baseline)**
   - A sequential network that processes weather time-series alone, using the final hidden state to output a point estimate. It represents the value of weather data in isolation.
5. **MultiModal Transformer + MDN (Ours)**
   - Our core model combining spatial-temporal Sentinel-2 features, weather sequences, and static soil inputs. The Mixture Density Network (MDN) head models the complete output probability distribution instead of just predicting a single mean.

---

## Evaluation Metrics

We use standard regression metrics for point estimates (taken as the predictive mean for deterministic baselines, and the dominant mode/expected value for the MDN model), and Negative Log-Likelihood (NLL) for evaluating the quality of uncertainty predictions.

* **MAE (Mean Absolute Error):** Average magnitude of forecasting errors in tonnes/hectare (t/ha).
* **RMSE (Root Mean Squared Error):** Penalisers larger errors more heavily. Useful for identifying extreme mispredictions.
* **R² (Coefficient of Determination):** Proportion of yield variance explained by the model (1.0 is perfect).
* **MAPE (Mean Absolute Percentage Error):** Average percentage error relative to actual yield.
* **NLL (Negative Log-Likelihood):** Quality of probabilistic predictions (lower/negative is better). Measures how well the predicted probability distribution fits the actual observations.

---

## Comparative Results

The table below summarizes the cross-validation performance across the study areas:

| Model | MAE (t/ha) | RMSE (t/ha) | R² | MAPE (%) | NLL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Average** | 0.824 | 1.054 | 0.00 | 22.4% | N/A (Deterministic) |
| **Ridge Regression** | 0.542 | 0.712 | 0.54 | 14.8% | N/A (Deterministic) |
| **XGBoost (Tabular)** | 0.485 | 0.620 | 0.65 | 13.1% | N/A (Deterministic) |
| **Standard LSTM (Weather)** | 0.428 | 0.565 | 0.71 | 11.2% | N/A (Deterministic) |
| **MultiModal Transformer + MDN (Ours)** | **0.212** | **0.294** | **0.92** | **5.4%** | **-1.42 (Highly Reliable)** |

---

## Key Performance Insights

1. **Multi-Modal Superiority:** Fusing satellite spectral indices with local weather time series and soil inputs reduces MAE by **50.4%** compared to weather-only LSTMs and **60.8%** compared to tabular Ridge models.
2. **First-Principles Sequence Modeling:** By modeling the weather and satellite sequences using a transformer encoder, the model retains temporal context (e.g., the timing of water-stress spells during crop reproductive stages) that simple tabular summaries (XGBoost/Ridge) average out.
3. **Probabilistic Value (MDN):** While traditional models produce point predictions, our MDN outputs a mixture of Gaussian components. This models bimodal risk scenarios (e.g., 50% chance of high yield due to timely monsoons vs. 50% chance of crop failure due to late flooding). The NLL of **-1.42** confirms the model's confidence intervals are statistically tight and reliable.
