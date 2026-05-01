# 🧠 BMI Classification App (ML + Streamlit)

A simple Machine Learning web app that predicts whether a person is **Underweight**, **Normal**, or **Overweight** based on **Weight** and **Height**.

---

## 🚀 Project Overview

This project uses a **K-Nearest Neighbors (KNN)** model trained on a small dataset to classify BMI categories.
The trained model is deployed using **Streamlit** as an interactive web application.

---

## 📊 Features

* Predict BMI category using:

  * Weight (kg)
  * Height (cm)
* Real-time predictions
* Simple and clean UI
* Model saved and loaded using `joblib`

---

## 🛠️ Tech Stack

* Python
* scikit-learn
* pandas
* numpy
* joblib
* streamlit

---

## 📁 Project Structure

```
├── deployment.py         # Streamlit app
├── knn_model.pkl         # Trained model
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bmi-classifier.git
cd bmi-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---


## 📈 Input Example

| Weight | Height |
| ------ | ------ |
| 70     | 175    |

---

## 🎯 Output

* Underweight
* Normal
* Overweight

---

## 🌐 Deployment

This app can be deployed using **Streamlit Community Cloud**:

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Select `deployment.py`
4. Deploy 🚀

---

## 📌 Future Improvements

* Add more data for better accuracy
* Include BMI formula-based validation
* Improve UI/UX
* Add visualization charts

---


