# 🏋️‍♀️ Fitness AI Agent

A smart Fitness AI Agent that provides **personalized exercise and diet plans** based on user details such as age, gender, BMI, fitness goal, and health conditions.

---

## 🚀 Features

- 💬 Conversational AI fitness assistant
- 📊 BMI calculation & category detection
- 🏃 Personalized exercise recommendations
- 🥗 ML-based diet plan prediction
- 🩺 Considers **Age, Gender, Hypertension & Diabetes**
- 🌐 FastAPI backend
- 📱 Mobile-friendly web UI

---

## 🛠️ Tech Stack

**Backend**
- Python
- FastAPI
- Scikit-learn
- Pandas
- Uvicorn

**Frontend**
- HTML5
- CSS3
- JavaScript

---

## ▶️ How to Run the Project (Step-by-Step)

Follow these steps to run the **Fitness AI Agent** on your local machine.



### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/Thiyumi2003/fitness-ai-agent.git
cd fitness-ai-agent 
```
### 🔹 Step 2: Create a Virtual Environment
```bash
python -m venv venv
```
### 🔹 Step 3: Activate Virtual Environment
**Windows**
```bash
venv\Scripts\activate
```
**Mac / Linux**
```bash
source venv/bin/activate
```
### 🔹 Step 4: Install Required Packages
```bash
pip install -r requirements.txt
```
### 🔹 Step 5: Run the FastAPI Server
```bash
uvicorn backend.main:app --reload
```
### 🔹 Step 6: Open in Browser
```bash
http://127.0.0.1:8000
```

