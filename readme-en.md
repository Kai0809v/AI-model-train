# 🌬️ CEEMDAN-LightGBM-Transformer Wind Power Intelligent Prediction System
[🇨🇳 中文版](./readme.md) | 🇺🇸 English Version
<div align="center">
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
**High-precision Wind Power Prediction System Based on Hybrid Deep Learning Architecture**
</div>

---

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Instructions](#usage-instructions)
- [Model Training](#model-training)
- [Performance Metrics](#performance-metrics)
- [FAQ](#faq)
- [Secondary Development](#secondary-development)
- [License](#license)

---

## 🎯 Project Overview
This is a lightweight, industrial-grade wind power prediction system based on the **CEEMDAN-LightGBM-Transformer** hybrid architecture. It provides a visual GUI interface, supports single-step/multi-step rolling prediction, and has built-in physical rule verification. It is suitable for wind farm power generation planning, power grid dispatching and other core scenarios.

---

## ✨ Core Features
- **High-precision Prediction**: CEEMDAN denoising + Transformer attention mechanism to improve time series prediction accuracy
- **Physical Rule Verification**: Cut-in wind speed interception and negative power correction, conforming to actual power generation laws
- **Easy to Use**: Visual GUI interface, usable without programming basics
- **Flexible Deployment**: Supports both GUI interaction and API calls for different scenarios
- **Lightweight & Efficient**: Install dependencies on demand, no full deployment required

---

## 📦 Installation Guide
### System Requirements
- OS: Windows 10/11, Linux, macOS
- Python Version: 3.8 or higher
- GPU Acceleration (Optional): NVIDIA CUDA 11.0+

---
💡 **Tip**
Regular Users: Only install `torch, numpy, pandas, matplotlib, scikit-learn, joblib, PySide6` to run the GUI for prediction
Researchers: If you need to retrain the model or modify the algorithm, please additionally install `PyEMD` and `lightgbm`
---

### Install Dependencies On Demand (No Need to Install All Libraries)
#### Scenario 1: Only Use GUI/API for Prediction (Recommended for Regular Users)
Only install **runtime essential dependencies**, no training libraries required:
```bash
# Step 1: Create and activate a virtual environment (Recommended)
conda create -n wind_forecast python=3.9
conda activate wind_forecast

# Step 2: Install core runtime dependencies
pip install torch numpy pandas matplotlib scikit-learn joblib PySide6
```

#### Scenario 2: Model Training / Secondary Development (For Researchers)
Based on Scenario 1, **additionally install training/development dependencies**:
```bash
# Install extra training-specific libraries
pip install PyEMD lightgbm
```

### Verify Installation
```bash
# Verify basic dependencies (Regular Users)
python -c "import torch, pandas, PySide6; print('Basic dependencies installed successfully')"

# Verify full dependencies (Developers/Training Users)
python -c "import PyEMD, lightgbm; print('Full dependencies installed successfully')"
```

---

## 🚀 Quick Start
### Method 1: Run GUI Interface (Best for Regular Users)
```bash
python GUI.py
```
Operation Steps:
1. Log in directly after startup (no registration required, any username/password)
2. Select a CSV historical data file that meets the format requirements
3. Select prediction steps (1-step/1-hour/2-hours/4-hours)
4. Click **Start Intelligent Prediction** and view visual results

### Method 2: Call API Interface (For Developers)

```python
from api_v6 import ForecastService
import pandas as pd

# Initialize prediction service
service = ForecastService(base_models_dir="assets")

# Load data
df = pd.read_csv("your_data.csv")

# Single-step prediction
result = service.run("CEEMDAN_LGBM_Transformer", df, steps=1)
print(f"Predicted Power: {result['prediction']:.2f} MW")

# Multi-step prediction (Next 4 steps)
result = service.run("CEEMDAN_LGBM_Transformer", df, steps=4)
print(f"Next 4-step Prediction: {result['predictions']}")
```

---

## 📁 Project Structure
```
CEEMDAN/
├── GUI.py                # Graphical User Interface (Entry for Regular Users)
├── api_v5.py             # Prediction Service API (Entry for Developers)
├── train/                # Model Training Module (Only for Development/Training)
│   ├── part1_v6_stable.py # Data Preprocessing + CEEMDAN Denoising
│   └── part2_v6_stable.py # Transformer Model Training
├── pretrained/           # Pretrained Model Repository (No Modification Needed)
└── res/                  # GUI Resource Files (No Modification Needed)
```

---

## 📖 Usage Instructions
### Data Format Requirements
The CSV file must contain the following columns (**exact column names**):

| Column Name | Description | Unit |
|------|------|------|
| Wind Speed at 10m (m/s) | Wind speed at 10m height | m/s |
| Wind Speed at 30m (m/s) | Wind speed at 30m height | m/s |
| Wind Speed at 50m (m/s) | Wind speed at 50m height | m/s |
| Wind Speed at 70m (m/s) | Wind speed at 70m height | m/s |
| Wind Speed at Hub Height (m/s) | Wind speed at wind turbine hub height | m/s |
| Wind Direction at 10m (°) | Wind direction at 10m height | ° |
| Wind Direction at 30m (°) | Wind direction at 30m height | ° |
| Wind Direction at 50m (°) | Wind direction at 50m height | ° |
| Wind Direction at 70m (°) | Wind direction at 70m height | ° |
| Wind Direction at Hub Height (°) | Wind direction at wind turbine hub height | ° |
| Temperature (°) | Ambient temperature | °C |
| Air Pressure (hPa) | Atmospheric pressure | hPa |
| Humidity (%) | Relative humidity | % |
| Actual Power Output (mw) | Historical active power | MW |

### GUI Operation Flow
1. Login: Enter any username/password and click Login
2. Select Data: Click **Browse** to choose a formatted CSV file
3. Configure Parameters: Select prediction scenario, algorithm model, prediction steps
4. View Results: Check power value on the left, prediction curve in the middle, system logs at the bottom

---

## 🎓 Model Training
Only execute the following steps when retraining or tuning the model:

### Preparation
Prepare a formatted historical data file `wind_data.csv` and place it in the project root directory.

### Step 1: Data Preprocessing
```bash
cd train
python part1_v6_stable.py
```
Functions: CEEMDAN denoising, feature derivation, data standardization, sliding window slicing, output training/test sets.

### Step 2: Model Training
```bash
python part2_v6_stable.py
```
Functions: LightGBM feature selection, Transformer model training, early stopping optimization, save pretrained model to `pretrained/` directory.

### Adjust Hyperparameters
Modify key parameters in `train/part1_v6_stable.py` or `part2_v6_stable.py`:
```python
# part1_v6_stable.py
WINDOW_SIZE = 96       # Historical window length (24 hours)
SPLIT_RATE = 0.9       # Train/Test split ratio

# part2_v6_stable.py
batch_size = 64        # Batch size
epochs = 50            # Maximum training epochs
early_stop_patience = 15  # Early stopping patience
```

---

## 📊 Performance Metrics
Core performance on real noisy datasets:
| Metric | Value | Description |
|------|------|------|
| RMSE | 6.2446 MW | Root Mean Square Error |
| MAE | 3.67633 MW | Mean Absolute Error |
| R² | 0.9848 | R-squared (higher = more accurate) |

- After CEEMDAN denoising, R² is improved by 5.4% compared with non-denoised baseline
- Physical rule verification ensures low-wind/negative power predictions conform to reality

---

## ❓ FAQ
### Q1: "No module named 'PyEMD'"
A: Only required for training/development: `pip install PyEMD` (if failed, try `pip install emd-signal`)

### Q2: GUI black screen/crash on startup
A:
1. Check PySide6 installation: `pip show PySide6`
2. Confirm `background.png` and `icon.png` exist in `res/` directory
3. Temporarily comment out background image code in GUI.py

### Q3: Negative prediction values
A: The API has built-in negative value correction (auto-set to 0), no extra handling needed.

### Q4: How to use custom datasets?
A: Ensure CSV column names match requirements, modify `DATA_PATH` in `part1_v6_stable.py`, and re-run training.

---

## 🔧 Secondary Development
### Extend New Algorithms
1. Define a new Predictor class in `api_v5.py` (refer to existing class structure)
2. Implement `predict()` and `predict_multi()` methods
3. Register the new model in `ForecastService._model_registry`
4. Train the model and save it to the `pretrained/` directory

### Modify Network Structure
Edit the `TransformerModel` class in `train/part2_v6_stable.py`:
```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)  # Adjust hidden layer dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8,  # Adjust attention heads
            dim_feedforward=512,   # Adjust feedforward network dimension
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # Adjust layers
```

### Custom Feature Engineering
Edit the `add_physics_features()` function in `train/part1_v6_stable.py` to add domain-specific features.

---

## 📜 License
This project is open-source under the MIT License. See the LICENSE file for details.

## 📮 Contact
For questions or suggestions, please contact:
- Email: zhouyukai.kevin@qq.com

<div align="center">
⭐ If this project helps you, please support us! ⭐
<br>
Made with ❤️ by Xichang University machine learning Wind Energy Research Team
</div>