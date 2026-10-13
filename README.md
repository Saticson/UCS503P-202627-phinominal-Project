# Sukio: A Marketplace for Machine Learning Solutions

![TIET Logo](assets/tiet-logo.svg)

**UCS503P: Software Engineering (Project) — 2026–27 ODD Semester**  
**Thapar Institute of Engineering and Technology, Patiala**  
**Department of Computer Science and Engineering**

**Project Team:**
- **Keshav Juneja** (Roll No: `1024030692`, Group: 3CSED)
- **Satwik Sinha** (Roll No: `1024030699`, Group: 3CSED)

**Lab Instructor:** Dr. Raghav B. Venkataramaiyer (`bv.raghav@thapar.edu`)

---

## Academic Context & Repository Layout

This repository serves as the official deliverable for the **UCS503P Software Engineering Project**. In accordance with the course specification, the repository is structured around incremental, testable delivery:

| Component | Path / Pointer | Description |
|---|---|---|
| **Project Proposal** | [`project-proposal/main.tex`](project-proposal/main.tex) | LaTeX specification of the Sukio marketplace concept, Time-to-Value heuristics, and multi-stage funnel. |
| **Prototype Stage Report** | [`project-report-prototype-stage/main.tex`](project-report-prototype-stage/main.tex) | Empirical validation of the core **Build** engine (AutoML platform), TTS benchmarks, and Playwright verification. |
| **Final Stage Report** | [`project-report-final/`](project-report-final/) | Final milestone deliverable covering the Buy and Hire marketplace modules (Upcoming). |
| **Team Journals** | [`journals/`](journals/) | Individual weekly activity journals tracking architectural progress and deliverables. |
| **Documentation** | [`docs/`](docs/) | Markdown technical documentation built and deployed via MkDocs CI/CD. |
| **Source Code** | [`code/`](code/) | Full-stack platform implementation (Java 21 Spring Boot control plane, Thymeleaf + HTMX UI, ML engine). |

### Core Project Heuristics (UCS503P Selection Criteria)
- **Primary Heuristic — Time-to-Value (TTV):** The platform is designed to produce testable value within seconds. Our primary empirical metric is **Time-to-Shortlist (TTS)**: the elapsed time from tabular data ingestion to receiving a ranked baseline model leaderboard (median TTS $\leq 3.95\text{s}$, well within the $\leq 60\text{s}$ target).
- **Contextual Relevance:** Bridges the gap for student developers and small engineering teams who need production-grade ML without hiring full-time specialists or building training infrastructure from scratch.
- **Engine Availability:** Built on top of mature, battle-tested engineering engines: Java 21 / Spring Boot 3 for the high-concurrency web control plane, Apache POI for spreadsheet parsing, and Scikit-Learn, Optuna, and SHAP for automated modeling and explainability.

---

## Executive Overview: The Sukio Build Engine

**Sukio** is an end-to-end machine learning platform and marketplace structured across three interconnected capabilities:
1. **Build (The AutoML Engine):** Enables users to upload structured data, inspect automated quality diagnostics, launch hyperparameter sweeps, inspect SHAP decision attributions, and run live inference sandbox predictions.
2. **Buy (Marketplace Catalog):** A verified repository of pre-packaged ML models and reusable pipelines (Final Stage).
3. **Hire (Expert Commissioning):** An escalation pathway connecting buyers with ML specialists for custom problem formulations (Final Stage).

The **Prototype Stage** focuses on the end-to-end realization and validation of the **Build** engine.

---

## Web Control Plane

Sukio features a unified, server-driven control plane built with **Java 21 (Spring Boot 3.3)**, **Thymeleaf**, and **HTMX**, styled under the technical **Axiom Obsidian** dark design system (`#0e0e10` base, violet accents `#d0bcff`).

![Sukio Web Control Plane](docs/gifs/00_frontend_control_plane.gif)

### Core Control Surfaces

| Control Surface | Live Preview | Focus & Key Interactions |
|---|---|---|
| **Platform Dashboard**<br>`/dashboard` | ![Dashboard Overview](docs/screenshots/01_dashboard_overview.png) | Real-time cluster health, active training jobs, registered models inventory, throughput metrics, and dataset catalog. |
| **Dataset Diagnostics**<br>`/datasets` | ![Dataset Diagnostics](docs/screenshots/02_dataset_diagnostics.png) | Automated schema inspection, composite quality scoring (0–100), missingness analysis, duplicate flagging, and target leakage risk detection. |
| **AutoML Training Studio**<br>`/training` | ![AutoML Training Studio](docs/screenshots/03_automl_training.png) | Multi-model candidate selection, Optuna trial budget controls, cross-validation parameters, and live HTMX progress polling. |
| **Inference & Explainability**<br>`/models` & `/predict` | ![Inference & Explainability](docs/screenshots/04_inference_explainability.png) | Ranked leaderboard, champion model badges, global/local SHAP feature rankings, and real-time prediction confidence gauge. |

---

## Pipeline Walkthroughs

### 1. Dataset Upload & Quality Profiling
Upload CSV or Excel files (`.xlsx`, `.xls`) or load one-click benchmarks (Iris, Titanic, California Housing, Diabetes). The engine extracts data schemas, calculates composite quality scores with letter grades, and detects data leakage hazards prior to model training.

![Dataset Upload and Profiling](docs/gifs/01_data_profiling.gif)

### 2. AutoML Training & Live Leaderboard
Configure multi-model experiments across Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, and Neural Networks. The platform auto-detects task type (classification vs. regression) and executes Optuna hyperparameter searches with live progress streaming via HTMX.

![AutoML Training and Leaderboard](docs/gifs/02_automl_training.gif)

### 3. SHAP Explainability & Inference Sandbox
Inspect trained candidate models through SHAP feature importance attributions. The real-time inference sandbox pre-fills feature values sampled from the training dataset profile, accepts manual parameter overrides, and returns instantaneous prediction verdicts with confidence probabilities.

![SHAP Explainability and Inference](docs/gifs/03_shap_inference.gif)

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Browser Client UI                       │
│        Thymeleaf 3 + HTMX 1.9 + Axiom Obsidian CSS        │
│   (Dashboard, Datasets, Training Studio, Models, Predict)  │
└───────────────────────────┬────────────────────────────────┘
                            │  HTTP / HTMX Lazy-Loading
┌───────────────────────────▼────────────────────────────────┐
│             Spring Boot 3 Control Plane (Port 8000)        │
│   • FrontendController  • DatasetController (Apache POI)   │
│   • TrainingController  • ModelController (SHAP Engine)    │
│   • Async TaskExecutor  • Spring Data JPA Repositories     │
└────────┬──────────────────────────────────┬────────────────┘
         │                                  │
┌────────▼──────────┐             ┌─────────▼────────────────┐
│    Database       │             │   Local Storage Filesystem│
│  H2 (Dev / Local) │             │   • /storage/datasets    │
│  PostgreSQL (Prod)│             │   • /storage/models      │
└───────────────────┘             │   • /storage/reports     │
                                  └──────────────────────────┘
         │                                  │
┌────────▼──────────────────────────────────▼────────────────┐
│              ML Execution & Training Engine                │
│   • Embedded Simulation Fallback (100% self-contained)    │
│   • External Python Microservice (Optuna, Scikit, SHAP)    │
│   • Authenticated Local GPU Training Agent (CLI mode)      │
└────────────────────────────────────────────────────────────┘
```

| Layer | Technology Stack | Purpose |
|---|---|---|
| **Control Plane UI** | Thymeleaf 3, HTMX 1.9, Obsidian CSS | Server-rendered, reactive dark-mode interface with zero client-side build step |
| **Backend API** | Java 21, Spring Boot 3.3.4, Spring Security | Multi-threaded control plane, file parsing, job queueing, and REST endpoints |
| **Data Processing** | Apache POI 5.2, Jackson 2.17 | Multi-format Excel/CSV tabular ingestion and metadata extraction |
| **AutoML Engine** | Scikit-Learn, Optuna 3.6, SHAP 0.45 | Bayesian hyperparameter optimization and explainability (or embedded fallback) |
| **Database** | H2 In-Memory (Dev) / PostgreSQL 16 (Prod) | Experiment metadata, dataset records, model metrics, and user profiles |
| **Quality Assurance**| Playwright Test Runner (Brave Browser) | 16 automated end-to-end and integration tests with presentation artifact generation |

---

## Execution Modes

Sukio supports two distinct training execution modes selectable from the training studio interface:

### 1. Remote Server Mode (Default)
Training jobs are queued and executed on the platform server. Spring Boot coordinates dataset preparation, invokes the AutoML engine, tracks Optuna iterations asynchronously, and streams progress directly to the browser via HTMX polling.

### 2. Local GPU Agent Mode
For large datasets or proprietary on-premise hardware:
1. Configure the experiment in the browser and select **Local** mode.
2. Download the authenticated agent script from the UI (`/api/training/local-agent/download`).
3. Run the agent on your local CUDA-enabled workstation:
   ```bash
   python local_agent.py --job-id <JOB_ID> --token <AUTH_TOKEN>
   ```
4. The local agent trains the model using local GPU acceleration and syncs metrics and artifacts back to Sukio over HTTPS.

---

## Quickstart & Verification

### Prerequisites
- **Java 21 LTS** installed (`java -version`)
- **Node.js 18+** installed (`node -v`)
- **Brave Browser** or **Chromium** (for automated visual tests)

### 1. Launch Backend Server
Clone the repository and run the pre-configured Maven wrapper using the in-memory local profile:

```powershell
cd code/backend-java
.\mvnw.cmd spring-boot:run "-Dspring-boot.run.profiles=local"
```

The application will start on port `8000`:
- **Web Interface:** `http://localhost:8000/`
- **Platform Dashboard:** `http://localhost:8000/dashboard`
- **REST Health Endpoint:** `http://localhost:8000/api/health`

### 2. Run Automated Playwright Tests in Brave
The project includes a comprehensive, presentation-grade test suite configured to run in visible (headed) mode with a human-paced delay:

```powershell
cd code/tests/playwright
npm run test:demo
```

### 3. View Visual Test Output Gallery
All test screenshots and presentation captures are aggregated into a local, untracked gallery:

```powershell
# Open the gallery directly in your browser:
start code/tests/playwright/test-output/gallery.html
```

---

## Empirical Prototype Evaluation

In accordance with the **Prototype Stage Report** ([`project-report-prototype-stage/main.tex`](project-report-prototype-stage/main.tex)), the platform underwent empirical benchmarking across standard ML datasets:

| Benchmark Dataset | Shape (Rows $\times$ Cols) | Problem Type | Target Metric | Time-to-Shortlist (TTS) |
|---|---|---|---|---|
| **Iris Species** | $150 \times 5$ | Classification | Accuracy: 94.2% | **2.4s** |
| **Titanic Survival** | $891 \times 12$ | Classification | ROC-AUC: 86.5% | **4.8s** |
| **Diabetes Progression** | $442 \times 11$ | Regression | $R^2$: 0.881 | **3.1s** |
| **California Housing** | $20,640 \times 9$ | Regression | RMSE: 0.524 | **18.2s** |

All benchmark sweeps achieved a median TTS of **3.95 seconds** ($\leq 60\text{s}$ target). The 16 automated Playwright tests passed with 100% reliability.

---

## Documentation & MkDocs CI/CD

Documentation is maintained in the [`docs/`](docs/) directory and compiled using MkDocs with Material theme. Every commit to `master` automatically triggers the GitHub Actions workflow [`.github/workflows/mkdocs.yml`](.github/workflows/mkdocs.yml) to deploy the updated documentation site to GitHub Pages.

---

## License

This project is submitted as an academic project for UCS503P at Thapar Institute of Engineering and Technology and is licensed under the [MIT License](LICENSE).
