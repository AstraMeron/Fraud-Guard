# 🛡️ Fraud-Guard: Production-Grade Fraud Detection
## Business Problem

In the modern financial landscape, transaction fraud costs billions annually and compromises user trust. Most institutions rely on outdated, rigid rules that fail to catch sophisticated behavioral shifts. **Fraud-Guard** addresses this by identifying high-risk transactions in real-time, balancing the need for security with the necessity of a seamless customer experience.

---

## 🚀 Solution Overview

**Fraud-Guard** is an end-to-end Machine Learning as a Service (MaaS) platform. It uses a Random Forest Ensemble architecture trained on engineered behavioral features like transaction velocity and account maturity.  

The system includes:
- A **Flask API** for real-time inference  
- A **Streamlit Dashboard** for business stakeholders to visualize risk  
- **SHAP explainability** to interpret model decisions  

---

## 📊 Key Results

- **99% Precision**  
  When the model flags fraud, it is almost certainly correct—minimizing false positives and customer friction.

- **8.5% Performance Boost**  
  The Random Forest ensemble increased the **AUC-PR score** from **0.82 (Baseline)** to **0.89**.

- **0.5s Latency**  
  Real-time IP-to-country mapping and prediction processing for immediate decision support.


## 🚀 Key Features

### **Task 1 & 2: Data Engineering & Modeling**
- **Production Structure:** Refactored from notebooks into a modular Python package (`src/` architecture).
- **Efficient Geolocation:** Implemented range-based IP-to-Country mapping using `pandas.merge_asof`, optimized for large datasets.
- **Feature Engineering:** - Transaction velocity (frequency per user).
  - Time-based features (Hour of day, Day of week).
  - Account maturity (Time since signup).
- **Imbalance Handling:** Applied **SMOTE** to handle class imbalance (Fraud vs. Legit), ensuring the model doesn't ignore minority fraud cases.

### **Task 3: Model Explainability**
- **Transparency:** Integrated **SHAP** and **LIME** to provide global and local transparency.
- **Visualization:** Exported feature importance plots to help stakeholders understand "why" a transaction was flagged.

### **Task 4: Model-as-a-Service (MaaS)**
- **Flask API:** A RESTful API providing real-time predictions.
- **Schema Alignment:** Robust preprocessing pipeline within the API to ensure incoming JSON data matches training feature names and order.
- **Containerization:** Ready-to-deploy `Dockerfile` for consistent environments.

### **Task 5: Interactive Dashboard**
- **Streamlit Frontend:** A professional dashboard for real-time fraud probing.
- **Business Impact:** Integrated metrics showing "Estimated Savings" and "Fraud Prevention" rates.
- **Interactive Visuals:** Dynamic Plotly charts for exploring fraud geography and SHAP explainability.

---

## 📂 Project Structure
```text
├── data/               # Raw and processed datasets
├── models/             # Serialized (.pkl) trained models
├── reports/            # SHAP/LIME visualization exports
├── src/                # Modular logic (Preprocessing, Features, Training)
├── tests/              # Integration tests (test_api.py)
├── dashboard.py        # Streamlit Frontend
├── serve_model.py      # Flask API Backend
├── requirements.txt    # Project dependencies
├── Dockerfile          # Containerization config
└── .gitignore          # Keeps data and env files out of Git
```
# 🛠️ Installation & Usage

### 1. Prerequisites
- **Python 3.9+**
- **Git**
- **Virtual Environment Tool** (venv)

### 2. Setup Environment
Clone the repository and set up your local environment to ensure all dependencies are isolated.

```bash
# Clone the repository
git clone [https://github.com/AstraMeron/Fraud-Guard](https://github.com/AstraMeron/Fraud-Guard)
cd Fraud-Guard

# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install all project dependencies
pip install -r requirements.txt
```
### 3. Running the Fraud-Guard System
The system operates using a **Client-Server architecture**. You must have the API (Backend) running for the Dashboard (Frontend) to function.

#### **Step A: Start the Backend (Flask API)**
Open a terminal and run the model server. This loads the Random Forest model and prepares the `/predict` endpoint.
```bash
python serve_model.py
```
#### **Step B: Start the Frontend (Streamlit Dashboard)**
Open a **second terminal**, activate your virtual environment, and launch the interactive UI. This dashboard acts as the consumer for your Flask API.
```bash
# Ensure your venv is active in this terminal too
streamlit run dashboard.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`. If it doesn't, you can manually navigate to that address.*

### 4. Running Integration Tests
To verify the API is processing features correctly—specifically checking for **schema alignment** and **feature parity**—run the automated test script:

```bash
python tests/test_api.py
```
## 📊 Business Impact & Insights
This dashboard translates complex ML metrics into actionable business intelligence for stakeholders:

* **Financial Protection:** Real-time identification of high-risk transactions to prevent chargebacks and fraud-related losses.
* **Operational Transparency:** Uses **SHAP** values to explain why a specific transaction was flagged, reducing "black-box" distrust for auditors.
* **Strategic Mapping:** Visualizes fraud geography to help security teams focus resources on high-risk regions.


## 🎥 Demo

Check out the interactive dashboard at **http://localhost:8501** to test real-time fraud probing!

---

## ⚙️ Technical Details

- **Data**  
  Sourced from financial transaction logs; preprocessed using `pandas.merge_asof` for IP-to-Country mapping and **SMOTE** to handle class imbalance (Fraud vs. Legit).

- **Model**  
  Random Forest Classifier with 100 estimators; optimized for high precision.

- **Evaluation**  
  Validated using **5-fold Stratified Cross-Validation** (Mean F1: 0.71) and **AUC-PR** to focus on the minority fraud class.

---

## 🔮 Future Improvements

- **Deep Learning**  
  Implement RNNs/LSTMs to analyze temporal sequences of user behavior.

- **CI/CD Integration**  
  Automate model retraining and deployment using GitHub Actions.

- **Cloud Scaling**  
  Deploy the API using Kubernetes to handle high-volume transaction bursts.

---

## 👤 Author

**Meron Tilahun**  
- LinkedIn: https://linkedin.com/in/meron-tilahun-3a17b324b  
- GitHub: https://github.com/AstraMeron
