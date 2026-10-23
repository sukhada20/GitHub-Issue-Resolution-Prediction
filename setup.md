# Setup & Execution Guide

This guide provides step-by-step instructions to set up, configure, and run the **GitHub Issue Resolution Prediction** web application and machine learning pipeline.

---

## 📋 Prerequisites

- **Python**: Version `3.10`, `3.11`, or `3.12` (recommended).
- **Package Manager**: Standard `pip` or [`uv`](https://github.com/astral-sh/uv) (fast package manager).
- **Operating System**: Windows, macOS, or Linux.

---

## 🚀 Quick Start (Running the Streamlit Web App)

Since pre-trained model artifacts are already included in the [`models/`](models/) directory, you can launch the interactive dashboard immediately after installing dependencies.

### Option A: Using Standard Python `venv` & `pip`

#### 1. Open Terminal in the Project Root
```powershell
cd "c:\Users\sukha\Downloads\GitHub Issue Resolution Prediction"
```

#### 2. Create and Activate Virtual Environment

- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(If script execution is disabled, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On Windows (Command Prompt - CMD):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Launch the Application
```bash
python -m streamlit run app.py
```

---

### Option B: Using `uv` (Fastest Setup)

If you have `uv` installed:

```powershell
# 1. Create a Python 3.12 virtual environment
uv venv --python 3.12

# 2. Install dependencies into the virtual environment
uv pip install -r requirements.txt --python .venv

# 3. Run the Streamlit app
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## 🌐 Accessing the Application

Once launched, Streamlit will provide local and network URLs:

- **Local Dashboard**: [http://localhost:8501](http://localhost:8501)
- **Network Dashboard**: `http://<your-local-ip>:8501`

Open [http://localhost:8501](http://localhost:8501) in your web browser to explore:
1. **Interactive Prediction Simulator**: Enter issue attributes and get instant resolution estimates, confidence intervals, and 7-day closure likelihood.
2. **Preset Scenarios**: Compare maintainer vs. newcomer issues, bug reports with code snippets, and weekend submissions.
3. **Model Benchmark Comparison**: View MAE, RMSE, MedAE, ROC-AUC, and calibration charts.
4. **KDD Methodology & Architecture**: Complete walkthrough of the 6 KDD phases.

---

## 🔄 (Optional) Re-running the Full KDD Pipeline

To re-process data, retrain models from scratch, or run tests:

### 1. Run Data Preprocessing
Cleans raw research data, parses UTC timestamps, and extracts target durations:
```powershell
python src/preprocessing.py
```
*Output: `data/processed/issues_clean.csv`*

### 2. Train Models & Generate Benchmarks
Trains Baseline, Ridge, Decision Tree, Random Forest, and XGBoost models:
```powershell
python src/train.py
```
*Outputs: Saved models and evaluation metrics in [`models/`](models/)*

### 3. Run Validation Test Suite
Executes automated tests validating preprocessing, feature engineering, and inference:
```powershell
python tests/test_pipeline.py
```

---

## 🛠️ Troubleshooting & Common Issues

### 1. Windows: `ImportError: DLL load failed ... An Application Control policy has blocked this file`
**Cause**: When files or environments exist within the `Downloads` directory, Windows Defender / App Control blocks binary `.pyd` / `.dll` libraries (like NumPy's `bit_generator`) marked with the "Mark of the Web".

**Fix**: Run this PowerShell command from the project root to unblock all files:
```powershell
Get-ChildItem -Recurse | Unblock-File
```

### 2. Windows: `Activate.ps1 cannot be loaded because running scripts is disabled`
**Cause**: PowerShell's default script execution policy is restricted.

**Fix**: Allow scripts for the current terminal session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Port `8501` is Already in Use
**Fix**: Specify a custom port when launching Streamlit:
```powershell
python -m streamlit run app.py --server.port 8502
```

---

## 📁 Key File References

- [`app.py`](app.py): Main interactive Streamlit application.
- [`requirements.txt`](requirements.txt): List of Python package dependencies.
- [`src/predict.py`](src/predict.py): Inference engine for model predictions.
- [`src/preprocessing.py`](src/preprocessing.py): Data cleaning and feature creation.
- [`src/train.py`](src/train.py): Model training and evaluation script.
- [`README.md`](README.md): Comprehensive project documentation and KDD research findings.
