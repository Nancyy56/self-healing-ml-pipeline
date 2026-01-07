# 🧠 Self-Healing Machine Learning Pipeline  
### End-to-End MLOps Capstone Project

This project implements a **self-healing machine learning system** that automatically detects data drift, conditionally retrains models, tracks experiments, manages model versions, and deploys the latest production model via a Dockerized FastAPI service.

It simulates how **real ML systems behave in production**, where data changes over time and models must adapt without manual intervention.

---

## 📌 Problem Statement

In real-world ML systems:
- Data distributions change over time (data drift)
- Models silently degrade
- Manual retraining is slow and error-prone
- Production models lack governance and traceability

**Goal of this project:**  
Build an automated pipeline that:
- Detects when incoming data has drifted
- Retrains only when necessary
- Compares new vs production models
- Promotes better models automatically
- Serves predictions reliably in production

---

## 🧠 What Does “Self-Healing” Mean Here?

> The pipeline automatically **monitors itself** and **fixes itself** when data drift occurs.

Self-healing behavior:
- No drift → do nothing (save compute)
- Drift detected → retrain model
- Worse model → reject
- Better model → promote to production
- Serving layer always loads **latest production model**

---

## 🏗️ Architecture Overview

            ┌───────────────┐
            │   New Data    │
            └───────┬───────┘
                    │
            ┌───────▼────────┐
            │ Drift Detection │
            └───────┬────────┘
        Drift < Thresh │ Drift > Thresh
              ❌       │        ✅
                      ▼
            ┌──────────────────┐
            │ Model Retraining │
            └─────────┬────────┘
                      ▼
            ┌──────────────────┐
            │ MLflow Tracking  │
            │ & Model Registry│
            └─────────┬────────┘
                      ▼
            ┌──────────────────┐
            │ Model Promotion  │
            │ (Production)    │
            └─────────┬────────┘
                      ▼
            ┌──────────────────┐
            │ FastAPI Serving  │
            │ (Dockerized)    │
            └──────────────────┘

---

## 🧰 Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| ML | Scikit-learn |
| Workflow Orchestration | Prefect |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Registry |
| API Serving | FastAPI, Uvicorn |
| Containerization | Docker |
| Version Control | Git & GitHub |


---

