# 🎓 Predicting Academic Achievement in Cambodian Higher Education

> A Multi-Model Machine Learning Analysis of Student Lifestyle and Behavioral Factors Influencing Academic Performance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-frontend-61DAFB?logo=react)](https://react.dev/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**Course:** ITM-390 Machine Learning — American University of Phnom Penh (AUPP)
**Advisor:** Dr. Kutha Pin · School of Digital Technologies

---

## 📌 Overview

This project develops a machine learning pipeline to predict the **cumulative GPA** of Cambodian university students and classify them into academic performance tiers (*Excellent / Good / Average / At-Risk*) using self-reported behavioral and lifestyle data.

We designed and distributed an **original bilingual (English–Khmer) survey** across 9+ major Cambodian universities (RUPP, ITC, CADT, AUPP, KIT, Setec, UC, Paragon, Cam-ED), collecting a dataset that does not currently exist anywhere — the **first of its kind for Cambodian higher education**.

The final system includes:
- Trained regression & classification ML models
- A rule-based early warning system for at-risk students (predicted GPA < 2.0)
- A student-facing **Habit Mirror** web app with live GPA prediction
- An instructor dashboard with at-risk alerts and one-click PDF export

---

## 👥 Team

| Role | Name | Student ID | Skills |
|---|---|---|---|
| Team Leader · ML Pipeline · Modeling | Art Oudom | 2025577 | Python, Pandas, NumPy, Scikit-learn |
| Web App · Backend · Inference | Hok Singparady | 2024568 | Python, Computer Vision, Web Dev |
| Frontend · UI/UX | Long Visal | 2024552 | React.js, JavaScript, Tailwind CSS |
| Data Analysis | Chhay Soklivin | 2024545 | Python, Web Dev |
| Documentation | Hun Leng | 2024579 | Python, Web Dev |

---

## 🗂️ Project Structure

```
Academic-Predictor-machine-based-on-performance-/
│
├── .vscode/                            # Editor/workspace settings
│
├── backend/                            # FastAPI inference & recommendation service
│   ├── __pycache__/
│   ├── models/                         # Serialized/trained model artifacts
│   ├── app.py                          # FastAPI app entry point
│   ├── feature_config.py               # Feature schema & config constants
│   ├── feature_engineering.py          # Feature transformation logic used at inference
│   ├── prediction.py                   # GPA / tier prediction logic
│   └── recommendations.py              # Habit-based recommendation logic
│
├── data/
│   ├── clean/
│   │   └── CambodianStudyHabitandPerformanceCleaned(1).csv
│   ├── preprocessed/
│   │   └── preprocessed_data(1).csv
│   └── raw_data/
│       ├── cambodian_student_Study_habit_and_thier_performance - Academic_Predictor_Extended.csv
│       └── Cambodian_Students_habit_and_performance.csv
│── models/ 
│   ├── tier_classifier.pkl
│   └── tier_scaler.pkl
│   
│
├── frontend/                           # React + Tailwind student/instructor web app
│   ├── src/
│   │   ├── index.html
│   │   ├── input.css                   # Tailwind source stylesheet
│   │   └── output.css                  # Compiled Tailwind output
│   ├── .gitignore
│   ├── package.json
│   └── package-lock.json
│
├── notebook/                           # Exploratory & modeling notebooks
│   ├── data_inspect.ipynb
│   ├── data_cleaning.ipynb
│   ├── eda.ipynb
│   ├── preprocessed.ipynb
│   ├── regression_model.ipynb
│   ├── classification_model.ipynb
│   └── early_warning_system.ipynb
│
├── src/                                # Core Python ML pipeline (training-side)
│   ├── __pycache__/
│   ├── config.py                       # Paths, constants, hyperparameter grids
│   ├── data_loader.py                  # Data loading & train/test splitting
│   ├── models.py                       # Model definitions (regression & classification)
│   ├── tuning.py                       # GridSearchCV / hyperparameter tuning
│   ├── evaluation.py                   # R², RMSE, MAE, Accuracy, F1 metrics
│   ├── early_warning_system.py         # Rule-based GPA < 2.0 flagging logic
│   └── main.py                         # Pipeline orchestration entry point
│
├── .gitignore
├── README.md
└── requirements.txt                    # Python dependencies (ML + backend)
```

---

## 📊 Dataset

**Source:** Original bilingual (English–Khmer) self-collected survey
**Target sample:** 300–500 responses from 9+ Cambodian universities
**Survey link:** [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdm7Dtgzwja2H2GcT_GcC3Sz-fdmtxJr7OLkK051xtTKf1nNg/viewform)

### Features (24 total)

| Type | Features |
|---|---|
| **Numerical** | Age, Cumulative GPA *(target)*, Self-study hours/day, Weekly tutoring hours, Social media hours/day, Gaming hours/day, Exercise days/week, Sleep hours/night, Academic stress level (1–10) |
| **Categorical** | Gender, Hometown, University, Year of study, STEM major, Diet quality, Internet quality, Employment status, Study consistency, AI tool usage, Lecture attendance %, Commute duration, Study group frequency, Extracurricular activities, Academic tier *(target)* |

> ⚠️ **Privacy:** Raw survey responses are not included in this repository. The processed anonymized dataset is available under `data/clean/` and `data/preprocessed/`.

---

## 🔬 ML Pipeline

```
Raw Survey Data (data/raw_data/)
      │
      ├──► EDA (notebook/data_inspect.ipynb, notebook/eda.ipynb)
      │
      └──► Data Cleaning & Preprocessing
                (notebook/data_cleaning.ipynb, notebook/preprocessed.ipynb
                 → src/data_loader.py)
                      │
                      └──► Modeling (src/models.py, src/tuning.py)
                                ├──► Regression → Predict GPA (R², RMSE, MAE)
                                │      (notebook/regression_model.ipynb)
                                └──► Classification → Predict Tier (Accuracy, F1)
                                       (notebook/classification_model.ipynb)
                                          │
                                          └──► Evaluation (src/evaluation.py)
                                                    │
                                                    └──► Early Warning System
                                                         (src/early_warning_system.py)
```

### Models Trained

**Regression** (target: cumulative GPA)
- Linear Regression (baseline), Ridge, Lasso, Random Forest, Gradient Boosting, SVR

**Classification** (target: Excellent / Good / Average / At-Risk)
- Logistic Regression, Random Forest, Gradient Boosting, XGBoost, KNN

**Evaluation:** 80/20 train-test split · 5-fold cross-validation · GridSearchCV hyperparameter tuning

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Academic-Predictor-machine-based-on-performance-.git
cd Academic-Predictor-machine-based-on-performance-
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks (in order)

```bash
jupyter notebook notebook/
```

### 4. Run the training pipeline (optional, regenerates models)

```bash
python src/main.py
```

### 5. Start the backend API

```bash
cd backend
uvicorn app:app --reload
# API docs available at http://localhost:8000/docs
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 Web App — Habit Mirror

Students interact with a **chat-style survey interface** (like texting a study coach) while a **live GPA gauge** animates with every answer. On completion, they receive:
- A predicted GPA displayed on a **shareable profile card**
- A chart comparing their habits against the **top 10% of performers**

Instructors log in to a **separate dashboard** showing a live at-risk student table with one-click PDF export for intervention reports.

---

## 📈 Expected Results

| # | Expected Outcome |
|---|---|
| 1 | Best model achieves **R² ≥ 0.75** on test set |
| 2 | Top predictors: daily study hours, tutoring attendance, stress level, lecture attendance |
| 3 | Urban (Phnom Penh) vs. provincial students show distinct behavioral profiles |
| 4 | AI tool usage correlates positively with GPA among STEM students |
| 5 | Early warning system correctly flags **≥ 80%** of at-risk students (predicted GPA < 2.0) |

---

## 🗓️ Timeline

| Task | Wk 1 | Wk 2 | Wk 3 | Wk 4 | Wk 5 | Wk 6 | Wk 7 | Wk 8 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Literature Review & Survey Finalization | ◯ | ◯ | | | | | | |
| Survey Distribution (9+ universities) | | ◯ | ◯ | | | | | |
| Data Cleaning & Preprocessing | | | ◯ | ◯ | | | | |
| Feature Engineering & EDA | | | | ◯ | | | | |
| Model Training & Evaluation | | | | | ◯ | | | |
| Early Warning System | | | | | | ◯ | | |
| Web Development | | | | | ◯ | ◯ | ◯ | ◯ |
| Testing, Report Writing & Submission | | | | | | | ◯ | ◯ |

---

## 📚 References

1. A. Yadav, "Predicting Student Academic Performance," *IJAR*, vol. 13, no. 09, 2025. [DOI](http://dx.doi.org/10.21474/IJAR01/21706)
2. D. Bou, "Challenges and suggestions for improving the quality of higher education in Cambodia," *CJER*, vol. 5, no. 1, 2025. [DOI](https://doi.org/10.62037/cjer.2025.05.01.03)
3. T. Boravin, "Bridging the Digital Divide," CDRI Cambodia Outlook Conference, 2025. [Link](https://coc2025.cdri.org.kh/blogs/bridging-the-digital-divide-advancing-inclusive-and-equitable-education-in-cambodia)
4. K. Heng & K. Sol, "Challenges facing higher education in Cambodia," *CJER*, vol. 3, no. 1, 2023. [DOI](https://doi.org/10.62037/cjer.2023.03.01.01)
5. A. Baba & K. Bunji, "Prediction of Mental Health Problem Using Annual Student Health Survey," *JMIR Mental Health*, 2023. [DOI](https://doi.org/10.2196/42420)
6. H. Martires, "Students' engagement in social media in Cambodia," *Conhecimento & Diversidade*, 2019. [ResearchGate](https://www.researchgate.net/publication/338414910)
7. A. S. Dessauvagie et al., "Mental health of university students in Southeastern Asia," *Asia Pacific Journal of Public Health*, 2021. [DOI](https://doi.org/10.1177/10105395211055545)
8. A. L. A. D. Urs & A. Sudharshan P. K., "A multi-factor machine learning framework for predicting student academic performance," *MethodsX*, 2025. [DOI](https://doi.org/10.1016/j.mex.2025.103673)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*American University of Phnom Penh · School of Digital Technologies · ITM-390: Machine Learning*