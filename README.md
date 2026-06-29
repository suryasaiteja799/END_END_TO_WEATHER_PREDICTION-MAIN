# 🌦 Weather Prediction System using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview

This project is an end-to-end **Machine Learning Weather Prediction System** built using **Python**, **Scikit-learn**, and **Streamlit**.

The application predicts the weather condition based on meteorological parameters such as:

* 🌧 Precipitation
* 🌡 Maximum Temperature
* 🌡 Minimum Temperature
* 💨 Wind Speed
* 📅 Year
* 📅 Month
* 📅 Day
* 📅 Day of Week

The project compares two popular Machine Learning algorithms:

* 🌳 Decision Tree Classifier
* 🌲 Random Forest Classifier

Users can select either algorithm from the Streamlit interface and compare prediction results.

---

# 🚀 Features

* Complete Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Label Encoding
* Feature Scaling
* Decision Tree Classifier
* Random Forest Classifier
* Model Comparison
* Confusion Matrix
* Classification Report
* Accuracy Score
* Streamlit Web Application
* Prediction Confidence Score
* Professional User Interface

---

# 📂 Project Structure

```text
Weather_Prediction/
│
├── data/
│   └── seattle-weather.csv
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset Information

**Dataset Name**

Seattle Weather Dataset

**Number of Records**

1461

**Features**

| Feature       | Description         |
| ------------- | ------------------- |
| date          | Date                |
| precipitation | Rainfall            |
| temp_max      | Maximum Temperature |
| temp_min      | Minimum Temperature |
| wind          | Wind Speed          |
| weather       | Target Variable     |

Target Variable

```text
weather
```

Possible Classes

```text
drizzle
fog
rain
snow
sun
```

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

# 📈 Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Label Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Decision Tree Model
      │
      ▼
Random Forest Model
      │
      ▼
Model Comparison
      │
      ▼
Save Pickle Files
      │
      ▼
Deploy with Streamlit
```

---

# 🌳 Machine Learning Algorithms

## 1. Decision Tree

Decision Tree is a supervised learning algorithm used for classification and regression. It creates a tree-like model by splitting the dataset into branches based on feature values.

### Advantages

* Easy to understand
* Fast prediction
* Handles numerical and categorical data
* Easy visualization

---

## 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting.

### Advantages

* Higher accuracy
* Reduces overfitting
* Robust to noisy data
* Handles large datasets efficiently

---

# 📊 Model Evaluation

Evaluation Metrics

* Accuracy Score
* Classification Report
* Confusion Matrix

Example Output

```text
Decision Tree Accuracy : 0.87

Random Forest Accuracy : 0.92

Best Model : Random Forest
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Weather_Prediction.git
```

Go to project folder

```bash
cd Weather_Prediction
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Train the Models

Run

```bash
python train_model.py
```

Generated files

```text
models/

decision_tree_model.pkl

random_forest_model.pkl

scaler.pkl

label_encoder.pkl
```

---

# ▶ Run Streamlit

```bash
streamlit run app.py
```

Application URL

```text
http://localhost:8501
```

---

# 🌐 Streamlit Features

The application allows users to:

* Select Decision Tree or Random Forest
* Enter weather parameters
* Predict weather conditions
* Display prediction confidence
* View prediction summary
* Compare algorithms

---

# 📦 Requirements

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install using

```bash
pip install -r requirements.txt
```

---

# 📸 Screenshots

## Home Page

* Modern Streamlit Interface
* Sidebar Input Panel
* Algorithm Selection

## Prediction Output

```text
Algorithm Used

Random Forest

Predicted Weather

☀️ SUN

Confidence Score

97.84%
```

---

# 🚀 Future Improvements

* XGBoost Classifier
* Gradient Boosting
* AdaBoost
* Hyperparameter Tuning
* Cross Validation
* Feature Importance Visualization
* Real-Time Weather API
* Docker Deployment
* AWS Deployment
* CI/CD Pipeline
* User Authentication
* Model Monitoring Dashboard

---

# 👨‍💻 Author

**Lakshman Ulli**

**Data Scientist | Machine Learning Engineer | AI Engineer**

### Skills

* Python
* Machine Learning
* Deep Learning
* Data Science
* Generative AI
* Streamlit
* Flask
* React

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Create a Pull Request

---

# ⭐ Support

If you like this project,

⭐ Star this repository

🍴 Fork it

🐞 Report issues

💡 Suggest improvements

---

# 📄 License

This project is licensed under the MIT License.

---

## Thank You ❤️

If this project helped you learn Machine Learning or Streamlit, consider giving it a ⭐ on GitHub.
