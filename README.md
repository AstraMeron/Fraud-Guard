# Fraud-Guard: Production-Grade Fraud Detection

## 🛡️ Project Overview
This project is a high-performance fraud detection system developed for the 10 Academy KAIM training. It identifies fraudulent transactions by analyzing user behavior, geolocation data, and transaction patterns.

## 🚀 Key Features (Task 1 Complete)
- **Production Structure:** Refactored from notebooks into a modular Python package (`src/` architecture).
- **Efficient Geolocation:** Implemented range-based IP-to-Country mapping using `pandas.merge_asof`, optimized for large datasets.
- **Feature Engineering:** - Transaction velocity (frequency per user).
  - Time-based features (Hour of day, Day of week).
  - Account maturity (Time since signup).
- **Imbalance Handling:** Applied **SMOTE** to handle class imbalance (Fraud vs. Legit), ensuring the model doesn't ignore minority fraud cases.

## 📂 Project Structure
```text
├── data/
│   └── raw/            # Original datasets (Fraud_Data, IpAddress, CreditCard)
├── src/
│   ├── preprocessing.py      # Cleaning and Geolocation mapping
│   ├── feature_engineering.py # Time and Velocity features
│   └── model_training.py      # SMOTE and imbalance handling
├── main.py             # Entry point to run the full pipeline
├── requirements.txt    # Project dependencies
└── .gitignore          # Keeps data and env files out of Git


# 🛠️ Installation & Usage

### 1. Prerequisites
- Python 3.9+
- Git

---

### 2. Setup Environment

Clone the repository and set up the virtual environment:

```bash
# Clone the repository
git clone https://github.com/AstraMeron/Fraud-Guard
```
```bash
cd Fraud-Guard
```
# Create virtual environment
```bash
python -m venv venv
```
# Activate the environment (Windows)
```bash
venv\Scripts\activate
```
# Activate the environment (Mac/Linux)
```bash
source venv/bin/activate
```