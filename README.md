# heartrate-forecasting


## Overview
This project implements a **Temporal Fusion Transformer (TFT)** model to predict heart rates of **ICU patients** up to six hours ahead. The model was trained on the **MIMIC-IV dataset** and demonstrated improved forecasting accuracy over **LSTM** and **GRU** models. The goal of this project is to **enhance early detection of patient deterioration**, supporting **proactive decision-making in ICU settings**.

## Features
- **Temporal Fusion Transformer (TFT) implementation** for multivariate time-series forecasting.
- **Comparative evaluation** against LSTM and GRU models.
- **Data preprocessing pipeline** for handling MIMIC-IV time-series data.
- **Visualization tools** for analyzing heart rate trends and model predictions.
- **Scalability** for deployment in real-world ICU monitoring systems.

## Dataset
- **MIMIC-IV**: A publicly available dataset containing ICU patient records.
- Features used: **Heart rate, vital signs, demographics, and medical history**.
- Preprocessing: **Missing value imputation, normalization, and time-alignment**.

## 🏗Model Architecture
- **Temporal Fusion Transformer (TFT)**
  - Handles both **static** and **time-dependent** covariates.
  - Utilizes **multi-head attention** for long-term dependencies.
  - Designed for **robust multivariate forecasting**.
