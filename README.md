# 🧠 SmartCrowd: Predicting Public Space Congestion Using BiLSTM

A Deep Learning project that predicts public space crowd congestion using a Bidirectional Long Short-Term Memory (BiLSTM) neural network. The model analyzes historical crowd patterns along with environmental and temporal factors to accurately forecast future crowd density.

---

## 📌 Project Overview

Crowd congestion prediction plays a crucial role in smart city management, event planning, transportation systems, and public safety.

This project leverages historical crowd data and contextual information such as weather, holidays, temperature, pollution levels, and special events to forecast future crowd counts using a BiLSTM model.

---

## 🎯 Objectives

- Predict future crowd density using historical time-series data.
- Improve congestion management through accurate forecasting.
- Analyze how environmental and temporal factors influence crowd movement.
- Demonstrate the effectiveness of Bidirectional LSTM networks for sequential prediction tasks.

---

## 📊 Dataset

The dataset contains hourly crowd information collected over one year.

### Dataset Information

- **Rows:** 8,760 (Hourly observations)
- **Columns:** 11

### Features

| Feature | Description |
|----------|-------------|
| timestamp | Date and Time |
| year | Year |
| month | Month |
| day_of_week | Day of Week |
| hour | Hour of Day |
| week_of_year | Week Number |
| is_holiday | Holiday Indicator |
| weather | Weather Condition |
| special_event | Event Indicator |
| temperature | Temperature |
| pollution_index | Pollution Level |
| crowd_count | Number of People (Target Variable) |

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🔍 Project Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Outlier Detection & Treatment
   │
   ▼
Data Preprocessing
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Data Scaling
   │
   ▼
Sequence Generation
   │
   ▼
BiLSTM Model
   │
   ▼
Training
   │
   ▼
Evaluation
   │
   ▼
Crowd Prediction
```

---

## 🧹 Data Preprocessing

The following preprocessing techniques were performed:

- Checked for missing values
- Outlier detection using boxplots
- Winsorization
- Robust Scaling
- Min-Max Normalization
- Time-series sequence generation
- Feature scaling
- Train, Validation, and Test split

---

## 📈 Exploratory Data Analysis

The notebook includes:

- Dataset overview
- Statistical summary
- Missing value analysis
- Boxplots
- Correlation Heatmap
- Crowd count over time
- Hourly crowd distribution
- Weekly crowd analysis

---

## ⚡ Feature Engineering

Additional features were created to improve model performance:

- Hour cyclic encoding (Sin & Cos)
- Day-of-week cyclic encoding
- Log transformation
- Robust scaled crowd count

---

## 🤖 Deep Learning Model

The project uses a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network consisting of:

- Bidirectional LSTM Layer (64 Units)
- Bidirectional LSTM Layer (32 Units)
- Dropout Layer
- Dense Hidden Layer
- Output Layer

The model was trained using:

- Adam Optimizer
- Mean Squared Error (MSE) Loss
- Mean Absolute Error (MAE) Metric
- Early Stopping
- Model Checkpoint

---

## 📊 Model Evaluation

Performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📁 Repository Structure

```
SmartCrowd-BiLSTM/
│
├── dataset/
│   └── synthetic_crowd_data_rich.csv
│
├── notebook/
│   └── SmartCrowd_Predicting_Public_Space_Congestion_Using_BiLSTM.ipynb
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SmartCrowd-BiLSTM.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## 💡 Applications

- Smart Cities
- Traffic Management
- Railway Stations
- Airports
- Shopping Malls
- Stadiums
- Metro Stations
- Event Management
- Public Safety
- Crowd Monitoring Systems

---

## 🔮 Future Improvements

- Real-time prediction using IoT sensors
- Integration with CCTV video analytics
- Weather API integration
- Streamlit Dashboard
- Model deployment using Flask or FastAPI
- Attention-based BiLSTM models
- Transformer-based time-series forecasting

---

## 👩‍💻 Author

**Charmmy Lalwani**

Computer Engineering Student

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
