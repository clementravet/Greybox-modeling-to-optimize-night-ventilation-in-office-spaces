# Grey-box modeling of office spaces to optimize nighttime ventilation for summer cooling

This repository contains the code developed for the Master's thesis:
> *Grey-box modeling of office spaces to optimize nighttime ventilation for summer cooling*
> **Clément Ravet** — supervised by Christian Anker Hviid, DTU Construct, DTU, Spring 2026

You can refer to the thesis report for full context, methodology, and results.

---

## Overview

`darkgreybox` implements grey-box thermal models of office spaces, combining physical 
structure with data-driven parameter estimation. The models are used to benchmark 
thermal behavior and evaluate nighttime ventilation strategies for passive summer cooling.

### Core Workflow

1. **Model definition** — Grey-box RC thermal models of varying complexity
2. **Parameter estimation** — Calibration against measured office data
3. **Simulation** — Forward simulation using calibrated models
4. **Benchmarking** — Comparison of model performance across configurations
5. **Night ventilation strategies** — Evaluation of control strategies for summer cooling

---

## Repository Structure
'''
darkgreybox/
├── darkgreybox/
│ └── models/ # Grey-box model definitions
├── data/ # Input data required for simulation and calibration
├── simulation/ # Scripts reproducing all thesis results
│ ├── model_summer/ # All the different summer models simulated on the same period
│ ├── model_winter/ # All the different winter models simulated on the same period
│ ├── model_R01.06_mcmc/ # Implementing same model on different periods from April to October 2025 on the room R01.06
│ ├── model_R05.07_mcmc/ # Implementing same model on different periods from April to October 2025 on the room R05.07
│ └── night_ventilation/ # Night ventilation strategy evaluation
└── requirements.txt
'''

Files "mcmc" are estimating parameters with mcmc method for each month while the "full" files are using the parameters estimated on the full period (from April to October) and then simulated on each separate month.


---

## Installation

```bash
pip install -r requirements.txt
```

**Supported Python versions:** 3.9, 3.10, 3.11, 3.12

---

## Usage

A typical workflow starts by importing the models, loading the data, running 
calibration, and then simulating:

```python
from darkgreybox.models import YourModel
import pandas as pd

# Load input data
data = pd.read_csv("data/office_data.csv", index_col=0, parse_dates=True)

# Instantiate and fit model
model = YourModel()
model.fit(data)

# Run simulation
results = model.predict(data)
```

To reproduce the full thesis results, run the scripts in the `simulation/` folder 
in order. Each script is self-contained and documented with comments explaining 
each step.

---

## Data

The `data/` folder contains the measured office environment data (temperature, 
CO₂, ventilation flow rates, outdoor conditions) used for model calibration and 
simulation. See the thesis report for a full description of the measurement setup 
and data processing pipeline.

---

## Thesis Reference

Ravet, C. (2026). *Grey-box modeling of office spaces to optimize nighttime 
ventilation for summer cooling*. Master's Thesis, DTU Construct, Technical 
University of Denmark. Supervisor: Christian Anker Hviid.

---

## Cite as

```bibtex
@mastersthesis{Ravet2026greybox,
  author  = {Clément Ravet},
  title   = {Grey-box modeling of office spaces to optimize nighttime 
             ventilation for summer cooling},
  school  = {Technical University of Denmark, DTU Construct},
  year    = {2026},
  month   = {June},
  note    = {Supervisor: Christian Anker Hviid}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file 
for details.