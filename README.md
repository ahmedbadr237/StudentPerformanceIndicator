# 📚 Student Performance Indicator

This project predicts students' math scores based on various factors, using multiple regression models and ensemble learning techniques.

## 🚀 Overview
The **Student Performance Indicator** helps educators and students estimate math scores based on:
- **Gender**
- **Race/Ethnicity**
- **Parental Level of Education**
- **Lunch Type**
- **Test Preparation Course**
- **Reading Score**

## 📊 Models Used
The following regression models were trained and compared:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

Each model was evaluated using:
- **Adjusted R² Score**
- **Mean Absolute Error (MAE)**

Ensemble models were fine-tuned using **hyperparameter optimization**, and the best-performing model was selected for deployment.

## 🎯 Deployment
The model is deployed using **Streamlit**. You can test the app here:  
🔗 **[Live Demo](https://studentperformancepred.streamlit.app/)**

## 📥 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/ahmedbadr237/StudentPerformanceIndicator.git
cd StudentPerformanceIndicator
pip install -r requirements.txt
streamlit run app.py
