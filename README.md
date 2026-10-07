# Linear Regression Health Costs Calculator

## Overview
Regression neural network that predicts annual medical insurance expenses from demographic and lifestyle features.

**freeCodeCamp:** Machine Learning with Python Certification — [certification page](https://www.freecodecamp.org/certification/fcc-ec5175fa-1c9f-4a8e-ae70-0f191f3a5883/machine-learning-with-python)

## What the project covers
- Loading the `insurance.csv` dataset
- One-hot encoding `sex`, `smoker`, `region` with `pd.get_dummies`
- 80/20 train-test split
- Feature standardization with `StandardScaler` (fit on train, transform test)
- Dense neural network: `64 → 64 → 1` (ReLU, ReLU, linear)
- Training 200 epochs with MAE loss, validation split 0.2
- Evaluating on the held-out test set and plotting predictions vs. true values

## Result
**Testing set MAE < 3500** — passed the freeCodeCamp challenge.

## How to run
Open `solution.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells.
Dependencies: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

