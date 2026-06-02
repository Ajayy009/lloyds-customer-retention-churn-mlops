# 🏦 Lloyds Bank Customer Retention & Churn Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random_Forest-43A047?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-2C8EBB?style=flat)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)


## 📌 Project Overview

This project delivers an end-to-end Machine Learning and MLOps solution for customer churn prediction using a Lloyds Banking Group customer dataset.

The objective is to identify customers at risk of churn and support proactive retention strategies through predictive analytics, experimentation, and production-ready deployment.

```mermaid
graph TD
    %% Define Styles for GitHub Themes
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef deploy fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% Nodes
    A[(Customer Dataset)]:::data
    B[EDA & Business Analysis]:::process
    C[Data Cleaning & Preprocessing]:::process
    D[Feature Engineering]:::process
    E[One-Hot Encoding]:::process
    F[Train/Test Split]:::process
    G[SMOTE Balancing]:::process
    
    subgraph Model_Phase [Model Development & Evaluation]
        H[Train Baseline Random Forest]:::model
        I[Train Advanced XGBoost]:::model
        J{Compare Evaluation Scores}:::model
        K[Select Best Model: XGBoost]:::model
    end

    L[MLflow Tracking]:::deploy
    M[Pickle/Joblib Serialization]:::deploy
    N[FastAPI Deployment]:::deploy
    O[Docker Containerization]:::deploy
    P([Real-Time Churn Prediction]):::data

    %% Connections
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J -- "XGBoost Outperforms RF" --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P 
```



## 📊 Dataset & Business Problem

Customer churn is one of the most critical challenges in banking.

Retaining an existing customer is significantly more cost-effective than acquiring a new one.

The dataset contains demographic, transactional, and behavioral information for 1,000 banking customers, along with their churn status.

### Target Variable

- Churn = 1 → Customer left the bank
- Churn = 0 → Customer retained

---

## 🔍 Exploratory Data Analysis (EDA)

Several business insights emerged during exploration:

### Digital Disengagement
Customers who churned showed lower login activity compared to retained customers, indicating digital disengagement as an early warning signal.

### Age-Based Risk Segments
Customer churn was concentrated within specific age groups, particularly middle-aged customers.

### Spending Behavior
Total expenditure emerged as a strong indicator of customer value and retention risk.

### Marital Status & Demographics
Demographic characteristics showed measurable differences in churn behavior and were retained for modeling.

### EDA Dashboard

![EDA Dashboard](assets/eda_dashboard.png)

---

## 🛠 Data Preprocessing & Feature Engineering

The following preprocessing pipeline was implemented:

### Data Cleaning

- Missing values handled appropriately
- Duplicate checks performed
- Outlier treatment applied where necessary

### Feature Selection

Removed:

- CustomerID
- Highly redundant variables

Retained:

- Behavioral variables
- Transactional variables
- Demographic variables

### Encoding

Applied One-Hot Encoding to:

- Gender
- Marital Status
- Income Level

---

## ⚖️ Handling Class Imbalance

The dataset exhibited significant class imbalance between churned and retained customers.

Strategies evaluated:

### Random Forest with Class Weights

Implemented cost-sensitive learning through custom class weights.

### SMOTE

Applied Synthetic Minority Oversampling Technique (SMOTE) to improve minority-class representation during training.

---

## 🤖 Model Development

### Model 1: Random Forest

Implemented:

- Stratified Train/Test Split
- Stratified Cross Validation
- GridSearchCV Hyperparameter Tuning
- Cost-Sensitive Learning
- Threshold Optimization

### Random Forest Feature Importance

Although XGBoost was selected as the final deployment model, Random Forest feature importance analysis provided valuable business insights into customer churn behaviour.

![Random Forest Feature Importance](assets/random_forest_feature_importance.png)

---

### Model 2: XGBoost

Implemented:

- Stratified Train/Test Split
- SMOTE Resampling
- StratifiedKFold Cross Validation
- GridSearchCV Hyperparameter Optimization
- Threshold Optimization

Best Parameters:

- Learning Rate: 0.1
- Max Depth: 5
- Estimators: 200
- Subsample: 0.8

---

## 📈 Model Comparison & Final Selection
| Model | Balanced Accuracy | Precision | Recall | ROC-AUC |
|---------|---------|---------|---------|---------|
| Random Forest | 48.67% | 0.19| 0.24 | 0.474|
| XGBoost | 51.90 %| 0.22 | 0.39 | 0.470 |

After comparing both models using Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices, and business utility, XGBoost was selected as the final deployment model.

### Final XGBoost Confusion Matrix

![XGBoost Confusion Matrix](assets/xgboost_confusion_matrix.png)

### XGBoost Feature Importance

![XGBoost Feature Importance](assets/xgboost_feature_importance.png)

---

## 🚀 MLOps & Deployment

The final model was operationalized through a complete MLOps workflow.

### MLflow

Used for:

- Experiment Tracking
- Run Management
- Reproducibility

#### MLflow Dashboard

![MLflow Dashboard](assets/mlflow_dashboard.png)

---

### FastAPI

Developed REST endpoints for real-time churn prediction.

Capabilities:

- JSON Input
- Probability Prediction
- Churn Risk Classification

#### Swagger Documentation

![FastAPI Swagger](assets/fastapi_swagger_prediction.png)

#### Live Prediction Output

![FastAPI Output](assets/fastapi_output.png)

---

### Docker

Containerized the entire application stack for consistent deployment across environments.

#### Docker Container

![Docker Container](assets/docker_container.png)

---

## 🧰 Technology Stack

### Machine Learning

- Random Forest
- XGBoost
- Scikit-Learn
- SMOTE

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### MLOps

- MLflow
- pickle/Joblib
- FastAPI
- Docker

---

## 🎯 Business Impact

The solution enables:

- Early identification of churn-risk customers
- Data-driven retention campaigns
- Improved customer engagement strategies
- Efficient allocation of retention resources

By combining machine learning with production deployment practices, the project demonstrates how predictive analytics can support customer retention initiatives in the banking sector.

---

## 🔮 Future Improvements

- SHAP Explainability
- Advanced Feature Engineering
- Cloud Deployment
- CI/CD Automation
- Real-Time Monitoring & Drift Detection
