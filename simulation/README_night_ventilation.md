# Master Thesis - Night Ventilation Strategies

This Jupyter Notebook contains the code, models, and analysis for evaluating night ventilation strategies as part of a Master's Thesis. 

## Notebook Structure

The notebook is organized into the following sections:

### 1. Importation
Sets up the environment and loads the necessary Python libraries and custom modules.

### 2. Plot the results
Generates the initial visualizations for the model evaluations:
* **Train results:** Model performance on the training dataset.
* **Test results:** Model performance on unseen testing data.
* **Plot the main results:** Key graphs summarizing the baseline findings.

### 3. NIGHT VENTILATION
The core section detailing the ventilation strategy, optimization, and control logic:
* **Analysis of the existing behaviour:** Evaluating current system performance.
* **SFP calculation:** Specific Fan Power calculations.
* **Functional description of ventilation:** The logic and rules governing the ventilation.
* **Setup:** Preparing the simulation environment.
* **Pre computation block:** Initial data processing steps.
* **Causal setup to train on past month and evaluate on current month:** The training and evaluation pipeline.
* **Night ventilation multi objective optimization:** Balancing different goals (e.g., cost vs. comfort).
* **Rolling MPC:** Implementing Model Predictive Control.
* **DMI code:** Integration of weather data (Danish Meteorological Institute).

### 4. Sensitivity analysis
* **COP:** Testing how sensitive the outcomes are to changes in the Coefficient of Performance.

## How to Use

1. Ensure you have the necessary data files and dependencies installed (like `pandas`, `matplotlib`, and any custom modules).
2. Open the notebook in your Python environment.
3. Run the cells sequentially from top to bottom to reproduce the analysis and plots.