![Header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,8,30&height=200&section=header&text=✈️%20Airline%20Delay%20Predictor&fontSize=38&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=ML-based%20flight%20delay%20prediction%20using%20Logistic%20Regression%20and%20Random%20Forest&descAlignY=55&descSize=15)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

> *Predict whether your flight will be delayed — before you even get to the airport.*

---

## 📌 Overview

Flight delays cost the US airline industry billions of dollars annually and affect millions of passengers. This project builds a machine learning pipeline to predict flight delays using historical airline data, applying classification techniques to surface the key factors that cause delays.

---

## 🧠 ML Approach

The model goes through two classification stages:

### Stage 1 — Classification (Delayed vs. On-Time)

- **Logistic Regression** — baseline binary classifier
- **Random Forest** — ensemble approach for improved accuracy and feature importance

### Stage 2 — Delay Duration Prediction

After classifying a flight as delayed, a regression model estimates the expected delay duration in minutes.

---

## 📊 Dataset

The model is trained on the **US Airline On-Time Performance** dataset, which includes:

| Feature | Description |
|---|---|
| `MONTH`, `DAY_OF_WEEK` | Temporal patterns in delays |
| `ORIGIN`, `DEST` | Airport-specific delay tendencies |
| `DEP_TIME`, `CRS_DEP_TIME` | Scheduled vs. actual departure |
| `CARRIER` | Airline-specific delay history |
| `DISTANCE` | Flight distance |
| `WEATHER_DELAY`, `NAS_DELAY` | Cause categorization |

---

## 🔑 Key Findings

- **Day of week** is a strong predictor — Friday and Sunday flights delay more
- **Carrier** matters significantly — some airlines have chronic delay patterns
- **Departure time** is critical — early morning flights are far more reliable
- **Origin airport** congestion is a leading predictor of downstream delays
- Random Forest outperformed Logistic Regression by ~8% on F1-score

---

## 🛠️ Tech Stack

- **Python 3** — Core language
- **Pandas, NumPy** — Data processing and feature engineering
- **scikit-learn** — ML models (Logistic Regression, Random Forest)
- **Matplotlib, Seaborn** — Visualizations
- **Jupyter Notebook** — Exploratory analysis and presentation

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/shreyneil/Airline_Delay_Predictor.git
cd Airline_Delay_Predictor
jupyter notebook
```

Open `Airline_Delay_Predictor.ipynb` and run all cells.

---

## 📁 Project Structure

```
Airline_Delay_Predictor/
├── Airline_Delay_Predictor.ipynb   # Main analysis notebook
├── data/                           # Dataset
├── models/                         # Serialized model outputs
└── README.md
```

The dataset is sourced from the [BTS On-Time Performance dataset](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations). Download and place in the `data/` folder before running.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | ~78% | 0.74 | 0.71 | 0.72 |
| Random Forest | ~86% | 0.83 | 0.80 | 0.81 |

---

## 👨‍💻 Author

**Shreyash Sharma** — PM2 @ ThoughtSpot · ML hobbyist

- [GitHub](https://github.com/shreyneil)
- [LinkedIn](https://www.linkedin.com/in/shreyash-sharma-b19918117/)

---

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,8,30&height=100&section=footer)

*Found this useful? Give it a ⭐ — it helps others discover it!*
