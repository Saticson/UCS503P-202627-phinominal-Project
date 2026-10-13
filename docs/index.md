![Tiet Logo](assets/tiet-logo.svg){ .tiet-logo }

**UCS503: Software Engineering (Project)**  
**TIET Patiala**

# Sukio: Production-Grade AutoML Platform

**Authors**:
- **Keshav Juneja** (Roll No: `1024030692`, CSED)
- **Satwik Sinha** (Roll No: `1024030699`, CSED)

**Course**: UCS503P (Software Engineering)  
**Instructor**: Dr. Raghav B. Venkataramaiyer  

---

## 1. Overview

**Sukio** is an enterprise-grade, full-stack AutoML platform engineered for dataset diagnostics, automated multi-model training, explainability (SHAP & LIME), and low-latency inference. 

Originally conceived as a Python (FastAPI) and Next.js control plane, the platform has been ported to **Java 21 (Spring Boot 3)** paired with **Thymeleaf + HTMX**, creating a hyper-responsive, server-driven web architecture following the **Axiom Obsidian** dark design system.

---

## 2. Architecture & Tech Stack

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚             Browser UI (Axiom Obsidian)                â”‚
â”‚       Thymeleaf Templates + HTMX + Tailwind CSS        â”‚
â”‚    (Dashboard, Datasets, Training, Models, Predict)    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚ HTTP / HTMX
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚          Spring Boot 3 Control Plane (Port 8000)       â”‚
â”‚   Controllers â”€â”€â–º Services â”€â”€â–º JPA Entities (H2/PG)    â”‚
â”‚   Async TaskExecutor â”€â”€â–º Embedded Simulation Engine    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚ REST (Optional)
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚      Python ML Engine (scikit-learn, Optuna, SHAP)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

- **Backend**: Java 21, Spring Boot 3.3.4, Spring Data JPA, Spring Security
- **Frontend**: Thymeleaf 3, HTMX 1.9.10, Axiom Obsidian CSS (`#0e0e10`, violet accents)
- **Database**: In-memory H2 (Local/Development mode) or PostgreSQL 16 (Production/Docker)
- **Testing**: Playwright End-to-End Test Suite (`code/tests/playwright/`)

---

## 3. Core Control Surfaces

1. **Platform Dashboard (`/dashboard`)**:
   Cluster metrics, registered models inventory, active background training runs, and GPU/engine health.
2. **Datasets & Diagnostics (`/datasets`)**:
   CSV and Excel upload with Apache POI, dataset summary statistics, sample preview tables, and data leakage detection.
3. **AutoML Training Studio (`/training`)**:
   Optuna Bayesian hyperparameter search over Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, and Neural Networks with live HTMX progress tracking.
4. **Model Leaderboard (`/models`)**:
   Ranked candidate models by validation score (Accuracy, F1-Score, ROC-AUC, RMSE) with SHAP feature attributions and artifact downloads.
5. **Inference Sandbox (`/predict`)**:
   Real-time prediction testing with arbitrary feature JSON payloads and confidence score distribution.

---

## 4. Running Locally

### Prerequisites
- Java 21 LTS
- Node.js 18+ (for Playwright tests)

### Build and Launch
From the repository root:

```bash
cd code/backend-java
./mvnw spring-boot:run -Dspring-boot.run.profiles=local
```

Access the control plane at: `http://localhost:8000/`

### Running Automated Playwright Tests
```bash
cd code/tests/playwright
npx playwright test
```
