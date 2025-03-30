# Perioperative Benchmark

A comprehensive machine learning framework for perioperative medicine research utilizing the INSPIRE dataset.

## Overview

This repository contains code for data preprocessing, feature engineering, and predictive modeling for perioperative medicine tasks. The project focuses on developing and evaluating machine learning models to predict various perioperative outcomes using electronic health record (EHR) data.

## Repository Structure

- **data_preprocessing/**: R scripts for data preparation and feature extraction
  - `main.R`: Main script to execute the preprocessing pipeline
  - `EMR_LIP.R`: Electronic Medical Record data processing
  - `0_operation.R` to `5_data_generation.R`: Sequential data processing scripts

- **modeling/**: Python modules for machine learning model implementation
  - `model.py`: Neural network model architectures (MLP, LSTM, GRU, BiLSTM, Attention models)
  - `dataloader.py`: Data loading utilities
  - `utils.py`: Helper functions for model training and evaluation
  - `utils_plot.py`: Visualization tools

- **Jupyter Notebooks**:
  - `preoperative_tasks_modeling.ipynb`: Models for preoperative outcome prediction
  - `intraoperative_tasks_modeling.ipynb`: Models for intraoperative outcome prediction
  - `postoperative_tasks_modeling.ipynb`: Models for postoperative outcome prediction
  - `metrics.ipynb`: Performance metrics calculation and analysis
  - `data_pivoting.ipynb`: Data transformation utilities
  - `miss_impute.ipynb`: Missing data imputation techniques
  - `scores_performance.ipynb`: Clinical scoring systems performance evaluation
  - `calculate_stats.ipynb`: Statistical analysis utilities

## Key Features

- Implementation of multiple neural network architectures including MLP, LSTM, GRU, BiLSTM with attention mechanisms
- Comprehensive data preprocessing pipeline for perioperative data
- Outcome prediction for various perioperative complications (stroke, AKI, ALI, etc.)
- Clinical scoring systems integration and evaluation
- Missing data handling strategies

## Requirements

- R (with packages: readxl, data.table, dplyr, magrittr, lubridate, stringr, mice, parallel, pbapply, jsonlite)
- Python 3.x
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- CUDA-capable GPU (recommended for training deep learning models)

## Usage

1. Data Preprocessing:
   ```
   Rscript data_preprocessing/main.R
   ```

2. Model Training and Evaluation:
   Execute the respective Jupyter notebooks for the perioperative phase of interest:
   - `preoperative_tasks_modeling.ipynb`
   - `intraoperative_tasks_modeling.ipynb`
   - `postoperative_tasks_modeling.ipynb`

## Data

This project utilizes the INSPIRE dataset, a publicly available research dataset for perioperative medicine. Due to data privacy concerns, the raw data is not included in this repository.

## Citation

If you use this code for your research, please cite:

```
@article{perioperative_benchmark,
  title={A Machine Learning Framework for Perioperative Outcome Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]},
  volume={[Volume]},
  pages={[Pages]}
}
```

## License

[Insert License Information] 