# Fraud Shell Company Detection

A small demo project that generates synthetic company records, compares
classification models, and provides a Streamlit interface for scoring a
company from its numerical features.

## Project files

- `eda.py` creates the synthetic dataset and exploratory visualizations.
- `train.py` compares Logistic Regression, Random Forest, and XGBoost, tunes a
  Random Forest, and saves the trained model and scaler.
- `str db.py` runs the Streamlit prediction interface.
- `shell_company_data.csv`, `best_fraud_model.pkl`, and `scaler.pkl` are
  generated data/model artifacts included for convenience.
- `Figure_1.png`, `heatmap.png`, and `Screenshot 2026-10-13 181143.png` are
  visualization and training output images.

## Setup

Use Python 3.10 or newer. From the project directory, create and activate a
virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

To generate the data and EDA plots:

```powershell
python eda.py
```

To train the models and regenerate the saved model and scaler:

```powershell
python train.py
```

To launch the prediction app:

```powershell
python -m streamlit run "str db.py"
```

## Note

This is an educational demonstration using synthetically generated data. The
reported model scores are not evidence of real-world fraud-detection
performance, and predictions should not be used to make real business or
compliance decisions.
