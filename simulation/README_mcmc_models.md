# Master Thesis: Interior Temperature Model (R01.06)

This Jupyter Notebook demonstrates the setup, training, and evaluation of a thermal grey-box model for interior temperature prediction as part of a Master's Thesis. 

## Overview
The model evaluates building thermal dynamics using the `DarkGreyBox` modeling framework, fitted via the `DarkGreyFit` tool. 

## Model Details
* **Model Used:** `TiTmCn2R2C_summer_V9`
* **Data Source:** `2025-0X.csv`
* **Temporal Resolution:** 15-minute intervals
* **Data Splitting:** 80% training data, 20% test data (split sequentially to preserve historical time-series context)

## Methodology
* **Fitting Method:** Nelder-Mead minimization (via `darkgreyfit`) + MCMC
* **Parameter Adjustment:** Capacitance and resistance ranges were deliberately widened to prevent the optimizer from hitting bounds during estimation.

## Outputs
* The trained model, dataframes, and results are serialized and saved to `simulation_results_full_model_2025_0X.pkl`.
* The notebook concludes by generating visual plots of the training and testing results (including Ventilation and Solar Gains).

## How to Use
1. Ensure the `2025-0X.csv` dataset is located in the appropriate directory.
2. Make sure the custom `DarkGreyBox` module is installed or added to your Python path.
3. Run the notebook (`R01.06_full_2025_0X.ipynb`) sequentially to reproduce the parameter estimations and generate the final result plots.