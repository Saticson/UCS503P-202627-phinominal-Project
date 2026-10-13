# Sukio Codebase: Architecture & Implementation

This directory contains the complete source code implementation for the **Sukio AutoML Build Engine**, transitioning the platform to a high-performance, server-driven architecture built on **Java 21 (Spring Boot 3.3)**, **Thymeleaf**, and **HTMX**.

---

## Directory Structure

```
code/
├── backend-java/                  # Spring Boot 3 Control Plane (Java 21)
│   ├── src/main/java/             # Controllers, Services, JPA Entities, DTOs
│   │   └── com/axiomcloud/
│   │       ├── config/            # SecurityConfig, StorageProperties, WebConfig
│   │       ├── controller/        # FrontendController, DatasetController, TrainingController, etc.
│   │       ├── dto/               # Request/Response Data Transfer Objects
│   │       ├── model/             # JPA Entities (Dataset, Job, Model, Experiment)
│   │       ├── repository/        # Spring Data JPA Repositories
│   │       └── service/           # DatasetService, TrainingService, MlEngineClient
│   ├── src/main/resources/
│   │   ├── templates/             # Native Thymeleaf Templates (Obsidian Dark Theme)
│   │   │   ├── index.html         # Landing / Overview
│   │   │   ├── dashboard.html     # Real-time Metrics & Inventory
│   │   │   ├── datasets.html      # Tabular Upload & Quality Diagnostics
│   │   │   ├── training.html      # AutoML Training Studio & Budget Sliders
│   │   │   ├── models.html        # Leaderboard Table & SHAP Explainability
│   │   │   └── predict.html       # Inference Sandbox & Confidence Gauge
│   │   ├── application.yml        # Default production configuration (PostgreSQL)
│   │   └── application-local.yml  # In-memory H2 development configuration
│   ├── pom.xml                    # Maven build descriptor (Java 21, Spring Boot 3.3.4)
│   └── mvnw.cmd / mvnw            # Maven Wrapper scripts
├── tests/playwright/              # Presentation-Grade End-to-End Test Suite
│   ├── demo-e2e-journey.spec.ts   # Complete Data -> AutoML -> Predict workflow
│   ├── demo-datasets.spec.ts      # Benchmark ingestion & diagnostics assertions
│   ├── demo-training.spec.ts      # Multi-model selection & Optuna budget tests
│   ├── demo-models.spec.ts        # Leaderboard ranking & SHAP attributions
│   ├── demo-predict.spec.ts       # Live prediction sandbox payload verification
│   ├── demo-visual-capture.spec.ts# High-resolution control surface captures
│   ├── generate-gallery.js        # Builds test-output/gallery.html (32 media items)
│   ├── playwright.config.ts       # Configured for visible execution in Brave Browser
│   └── package.json               # Node.js dependencies (@playwright/test)
├── docs/                          # Architectural media assets
│   ├── gifs/                      # Animated walkthrough GIFs
│   └── screenshots/               # Control plane screenshots
├── storage/                       # Local filesystem persistence
│   ├── datasets/                  # Uploaded CSV & Excel datasets
│   ├── models/                    # Serialized model binaries & checkpoints
│   └── reports/                   # Generated quality diagnostics reports
└── DESIGN.md                      # Detailed architectural design decisions
```

---

## Control Plane Architecture

The control plane replaces the previous client-heavy React single-page application with a reactive, server-driven architecture:

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
│  H2 (Dev / Local) │             │   • storage/datasets     │
│  PostgreSQL (Prod)│             │   • storage/models       │
└───────────────────┘             │   • storage/reports      │
                                  └──────────────────────────┘
         │                                  │
┌────────▼──────────────────────────────────▼────────────────┐
│              ML Execution & Training Engine                │
│   • Embedded Simulation Fallback (100% self-contained)    │
│   • External Python Microservice (Optuna, Scikit, SHAP)    │
│   • Authenticated Local GPU Training Agent (CLI mode)      │
└────────────────────────────────────────────────────────────┘
```

### Key Design Pillars
1. **Zero Client-Side Build Step:** No Webpack, Vite, or Node.js bundle steps are required to run the web application. Thymeleaf handles server-side layout and HTMX delivers instantaneous partial updates.
2. **Unified Obsidian Dark Theme:** Strict palette using `#0e0e10` base, `#19191d` surface containers, `#25252b` elevated panels, and `#d0bcff` violet accents.
3. **Resilient Embedded Simulation:** `MlEngineClient` provides an embedded simulation fallback. If the external Python engine on port 8001 is offline, the Spring Boot platform simulates Optuna sweeps, SHAP attributions, and inference results with 100% functional fidelity.

---

## Running the Application Locally

### Development Mode (In-Memory H2, No Docker Required)
Run the application with the `local` profile:

```powershell
cd code/backend-java
.\mvnw.cmd spring-boot:run "-Dspring-boot.run.profiles=local"
```

The application runs on `http://localhost:8000/`. H2 database console is enabled at `http://localhost:8000/h2-console`.

### Production Mode (PostgreSQL 16)
Ensure PostgreSQL is running, then launch without the local profile:

```powershell
cd code/backend-java
.\mvnw.cmd spring-boot:run
```

---

## Playwright Automated Test Suite

The test suite in `tests/playwright/` is configured to run visibly in your local **Brave Browser** (`headless: false`) with human-paced `slowMo` delay:

```powershell
cd code/tests/playwright

# Run all demo tests and build the visual gallery:
npm run test:demo

# Open the visual test gallery in your browser:
start test-output/gallery.html
```

All generated test artifacts (`test-output/`, screenshots, recordings) remain **strictly untracked** in Git via `code/.gitignore`.
