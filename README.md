# Spacecraft Thruster ML

ML pipeline for spacecraft thruster thrust prediction using time series data with exogenous inputs.

**Task:** Predict thrust[t] given ton[t] and historical data.

## Project Structure

```
thruster/
├── data/
│   ├── dataset/          # Full dataset (SN01-SN24)
│   └── samples/          # Sample files for testing
├── src/
│   ├── data/             # Loading, cleaning, aggregation
│   ├── exploration/      # Statistics, visualizations
│   ├── modeling/         # Feature selection, models, evaluation
│   └── scripts/          # Pipeline entry points
└── outputs/              # Results, figures, models
```

## Midterm 1: Data Pipeline

Data collection, cleaning, and exploratory analysis.

**Run:**
```bash
python -m src.scripts.first_midterm
```

**Features:**
- Load and validate metadata + time series CSV files
- Clean data: handle missing values, outliers, duplicates
- Aggregate time series into statistical features
- Generate descriptive statistics and visualizations

**Options:** `--log-file`, `--save-cleaned-csv`, `--interactive`

## Midterm 2: Modeling

Feature selection and model comparison for thrust prediction.

**Run:**
```bash
python -m src.scripts.second_midterm
```

**Pipeline:**
1. **Feature Engineering:** Lag features from time series
2. **Feature Selection:** SelectKBest, PCA comparison
3. **Models:** Ridge Regression, Random Forest, HistGradientBoosting
4. **Evaluation:** MAE, RMSE, R² on test set (SN13-24)

**Examples:**
```bash
# Quick test with feature selection
python -m src.scripts.second_midterm --max-files 50 --feature-selection

# Save processed data to cache (slow first time)
python -m src.scripts.second_midterm --save-cache --models ridge --skip-viz

# Fast runs from cache
python -m src.scripts.second_midterm --use-cache --feature-selection
```

## Setup

**Requirements:** Python 3.11+

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e ".[dev]"
```

## Development

```bash
pre-commit install    # Setup hooks
ruff check .          # Lint
ruff format .         # Format
pyright               # Type check
```
