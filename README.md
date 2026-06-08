# MATRIX AI - Machine Learning Project Report

## 1. Methodology
The objective of this project is to develop predictive models across multiple domains: Material Property Prediction, Commodity Price Forecasting, Cross-Domain Correlation Analysis, and Element Price Forecasting for Inverse Design. The overall methodology is structured into the following pipelines:

1. **Data Ingestion & Preprocessing:** Data from 5 distinct datasets (DS1-DS5) is loaded and preprocessed. Missing values are imputed using forward-fill for time-series data or zero-filling where appropriate. Categorical variables are encoded using `LabelEncoder`.
2. **Feature Engineering:** Domain-specific features are extracted from structural materials data, as well as time-series technical and macroeconomic indicators for commodity modeling.
3. **Model Training:** Machine learning models (Gradient Boosting, XGBoost, Random Forests) are trained to map input features to continuous targets (regression) or binary labels (classification).
4. **Evaluation:** Models are evaluated on an 80/20 train-test split to assess generalisation.
5. **Inverse Design Preparation:** Preparing a search space using trained models to evaluate and recommend novel material compositions based on optimal properties and element costs.

---

## 2. Feature Engineering

Feature engineering strategies were tailored to the specific datasets and prediction targets:

### Material Properties (DS1)
- **Structural Features:** Extracted `n_elements`, `spacegroup_number`, `nsites`, and `volume_A3`.
- **Categorical Encoding:** Applied explicit `LabelEncoder` transformations to `crystal_system` and `category`.
- **Stability Label Formulation:** A custom `is_stable` target label and continuous `stability_score` were engineered based on boundary conditions: Formation energy < 0, energy above hull < 0.1, and positive bulk/shear moduli.
- **MQI Normalization:** A Material Quality Index (MQI) mapping was created, scaling bulk modulus, shear modulus, formation energy, density, melting point, and band gap using `MinMaxScaler`.

### Commodity Price Forecasting (DS2 & DS3)
- **Technical Indicators:** Computed simple moving averages (SMA 21, 63), Bollinger Bands z-scores, Relative Strength Index (RSI 14), MACD, and Momentum (10d, 21d).
- **Cross-Domain Features:** Merged external market signals including daily Material Quality Index (MQI), supply disruption probability, substitution elasticity, green premium, and Herfindahl index.
- **Trend Variables:** Calculated rolling trends (5d, 21d, 63d) for MQI and short-term volatility.

### Element Price Forecasting (DS5)
- **Autoregressive Lags:** Generated lagged price features for 1, 2, 3, 6, and 12-month periods to capture seasonal and historical price dependencies. 

---

## 3. Model Architecture

The predictive modeling engine leverages ensemble tree-based models, chosen for their robustness to outliers, lack of strict feature scaling requirements, and ability to capture non-linear interactions natively.

### 3.1 Material Property Prediction
- **Architecture:** `GradientBoostingRegressor` (or `XGBRegressor` if available).
- **Hyperparameters:** `n_estimators=300`, `max_depth=5` (or 7 for XGB), `learning_rate=0.05`.
- **Target Variables:** Separate models trained for 8 targets (Bulk Modulus, Shear Modulus, Band Gap, Density, Formation Energy, Melting Point, Poisson Ratio, Energy Above Hull).

### 3.2 Material Stability Classifier
- **Architecture:** `RandomForestClassifier`.
- **Hyperparameters:** `n_estimators=200`.
- **Purpose:** Classifies whether a given material structure is physically stable.

### 3.3 Commodity Price Forecasting
- **Architecture:** `GradientBoostingRegressor` (or `XGBRegressor`).
- **Hyperparameters:** `n_estimators=400`, `max_depth=5` (or 6 for XGB), `learning_rate=0.03`.
- **Strategy:** Multi-horizon forecasting modeling individual price jumps at 1-day, 7-day, and 30-day forecast horizons.

### 3.4 Element Price Forecasting
- **Architecture:** `RandomForestRegressor`.
- **Hyperparameters:** `n_estimators=200`.
- **Strategy:** Autoregressive time-series prediction utilizing historical lags.

---

## 4. Evaluation Metrics

Models were validated on a contiguous 20% holdout set to preserve time-series integrity (for commodities) and generalization capacity (for materials):

- **Mean Absolute Error (MAE):** Used to measure the average magnitude of absolute errors in predictions without weighting extreme outliers heavily.
- **Root Mean Squared Error (RMSE):** Selected to penalize large prediction errors heavily, ensuring models are reliable under extreme deviation scenarios (e.g., severe price shocks or anomalous material formations).
- **R-Squared ($R^2$):** Assesses the proportion of the variance in the dependent target variable that is predictable from the independent variables.
- **Accuracy Score:** Utilized exclusively for the Random Forest Material Stability Classifier to represent the absolute correctness of binary stability predictions.
