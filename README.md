# 💻 PriceLens AI

<p align="center">
  <b>End-to-End Laptop Price Prediction with Explainable AI + LLM Insights</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/XGBoost-Model-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Explainability-SHAP-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b?style=for-the-badge">
</p>

---

## 🚀 Overview

PriceLens AI is a complete machine learning system that predicts laptop prices and explains the reasoning behind each prediction.

Unlike typical ML projects, this focuses on:

* 📊 Feature-level explainability (SHAP)
* 🧠 Human-readable reasoning
* 🤖 AI-generated insights (LLM)

---

## 🧠 Problem Statement

Laptop pricing is influenced by multiple interacting features:

* RAM, CPU, Storage
* Display quality
* Brand positioning

Challenges:

* Messy dataset with mixed-format columns
* Hidden signals inside raw text features
* Lack of interpretability in predictions

---

## ⚙️ Approach

### 🔹 Feature Engineering

* Extracted structured data from messy columns
* Created features like SSD/HDD split, PPI, CPU/GPU brand

### 🔹 Model Building

Models tested:

* Linear Regression
* Random Forest
* Gradient Boosting
* XGBoost

Final model selected based on performance and generalization using HyperParameter Tuning and Cross Validation
 Best Models: Gradient Boosting and XGBoost 
Selected Model: Gradient Boosting
R² Score: ~0.89 

### 🔹 Explainable AI

* SHAP used for feature contributions
* Log predictions converted to real price
* Built feature impact visualization

### 🔹 LLM Integration

* Gemini API generates natural explanations
* Converts technical output into human insights

---

## 📊 Features

* 💰 Price prediction
* 📈 Feature impact visualization
* 📋 Top influencing factors
* 🧠 Rule-based explanation
* 🤖 AI explanation
* 🖥️ Interactive Streamlit UI

---

## 🧪 Tech Stack

* Python
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Pandas, NumPy
* Gemini API

---

## 📂 Project Structure

```bash
PriceLens_AI/
│
├── app.py
├── requirements.txt
│
├── artifacts/
│   ├── model.joblib
│   └── meta.joblib
│
├── data/
│   └── laptop_data.csv
│
├── notebooks/
│   └── laptop.ipynb
│
├── src/
│   ├── ml_core.py
│   ├── visualization.py
│   └── insights.py
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/PranayBothra/PriceLens-AI.git
cd PriceLens-AI
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

For deployment, use Streamlit Secrets.

---

## 🌐 Live Demo

https://pricelens-ai.streamlit.app/

---

## 📌 Key Learnings

* Feature engineering is often harder than modeling
* Explainability makes ML usable
* LLMs improve interpretability
* End-to-end systems are more valuable than standalone models

---
## 📜 License

MIT License
