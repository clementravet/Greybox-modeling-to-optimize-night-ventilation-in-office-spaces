# Master Thesis: Thermal Model V4 (Summer and Winter Period)

This Jupyter Notebook contains the setup, training, and evaluation of a thermal grey-box model for a summer/winter period as part of a Master's Thesis. 

## Overview
The model evaluates a building scenario with zero active heating using the `DarkGreyBox` modeling framework, fitted via the `DarkGreyFit` tool. The temporal resolution of the data and model is set to 5 minutes.

## Notebook Structure

The notebook is organized into the following main sections:

### 1. Introduction and Setup
* **Model Overview:** Demonstrates the use of `DarkGreyBox` models by fitting them with `DarkGreyFit`.
* **Case Study:** Focuses on a specific period.

### 2. Data Loading, Preprocessing, and Splitting
* **Data Source:** Ingests the `data_summer/winter.csv` dataset.
* **Variables:** Maps key features including indoor temperature (Ti), CO2 concentration (c), ambient temperature (Ta), solar irradiation (Ik), and supply temperature (Tsup).
* **Train/Test Split:** Uses an 80/20 chronological split to strictly preserve the time-series order.

### 3. Physical Constants and Model Parameters
* **Room Geometry:** Sets up the physical room dimensions (11 m² surface area, 29.7 m³ volume, 5.48 m² window area).
* **Initial Guesses:** Calculates initial physical guesses for air/thermal mass heat capacitance and internal/envelope thermal resistance.

### 4. Model Initialization and Fitting Methods
* **Optimizer:** Utilizes the Nelder-Mead optimization method (via `lmfit.minimize`).
* **Error Metric:** Minimizes the Root Mean Square Error (RMSE).
* **Model Class:** `TiTmCn2R2C_summer/winter`.

### 5. Training Results and Parameter Estimation
* Outputs a detailed table showing the estimated values, bounds, and variances for each model parameter.
* Generates a final diagnostic comparison between the calculated physical guesses and the optimizer's final estimated values.

## How to Use
1. Place the `data_summer/winter.csv` file in the same directory or adjust the data path in the notebook.
2. Ensure the custom `DarkGreyBox` module is installed or accessible in your Python system path.
3. Open the notebook (`model_summer/winter_VX.ipynb`) in Jupyter or your preferred environment.
4. Run the cells sequentially from top to bottom to train the model and generate the result metrics.