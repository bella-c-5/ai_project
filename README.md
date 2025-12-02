# Resume Pilot – AI Resume Analysis Tool

This repository contains my final project for **CSI-5130: Artificial Intelligence**.

Resume Pilot is a tool I developed to help users improve their resumes. It will prompt the user to attach their resume pdf file and analyzes it for them. In order to provide improvement suggestions and notes, Resume Pilot utilizes AI techniques, such as probabilistic classification, Markov Decision Process, Deep Q-Learning, Policy Gradient scoring, and a PyTorch self-attention module. The backend is built with FastAPI and the frontend is built with CSS, JS, and HTML. 

---

## Features

Resume Pilot processes any PDF resume that is uploaded and utilizes multiple AI models to generate notes:

### **1. Probabilistic Classification**
Predicts the resume’s most likely career field by utilizing logistic regression and a scikit-learn vectorizer. This helps the user determine how closely their information matches the properties of a job they might be applying for.

### **2. Markov Decision Process (MDP)**
Resume improvementa are modeled as sequential decisions with rewards. 
e.g., add, improve

### **3. Deep Q-Learning (DQN)**
New skills are recommended based on a vector representation of the resume.

### **4. Policy Gradient Score**
The resume receives a score by learning a probability-based reward model.

### **5. PyTorch Self-Attention**
Importance is highlighted across text tokens by utlizing a simplified transformer-style module.

### **6. Full-Stack Architecture**
- **Backend:** FastAPI (Python)
- **Frontend:** HTML, CSS, JavaScript

---

## Project Structure
ai_project/

- app.py
- ai_models/
-- probabilistic_model.py
-- resume_mdp_model.py
-- skill_dqn_agent.py
-- pg_score_model.py
-- resume_attention_module.py

- data/
--  Resume.csv # training dataset (not included in repo)
--  resumes_uploaded/ # user-uploaded resumes (ignored by Git)

- frontend/ # Frontend UI
-- index.html
-- styles.css
-- script.js

- .gitignore # .venv, uploaded resumes, etc. are not tracked

## Required Dataset (Download Manually)

This project uses a public resume dataset from Kaggle, but the size is 50+MB and cannot be included in Github.

### **You must manually download:**
**Resume.csv**

### **Download link:**
🔗 https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

After downloading, place it inside: data


---

## User Guide

Follow these steps to run the project locally.

### **1. Clone the repository**
git clone https://github.com/bella-c-5/ai_project.git
cd ai_project

### **2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   (Personally using MacOS)

### **3. Install packages
pip install fastapi uvicorn pdfminer.six pandas scikit-learn pillow torch python-multipart spacy

### **4. Run FastAPI backend
uvicorn app:app --reload

### **5. Click server link

## Usage

1. **Upload your resume** (`.pdf` format only).

3. Click **Analyze**.

4. Read the results:

   - **Extracted Personal Information**  
   - **Skill Analysis**  
   - **Career Field Prediction**
   - **MDP-Based Improvement Sequence**  
   - **DQN Skill Recommendation**  
   - **Policy Gradient Resume Score**  
   - **Self-Attention Importance Vector** (PyTorch Attention Module)








