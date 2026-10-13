# System Architecture & Porting Guide

Sukio represents the foundational automated baseline engine within the **Sukio** machine learning solutions marketplace. This document describes the architecture resulting from the port from Python/Next.js to **Spring Boot 3 + Thymeleaf + HTMX**.

---

## 1. Architectural Motivation

The original architecture consisted of:
- **Frontend**: Next.js 14 client-rendered Single-Page Application (SPA) with complex client hydration.
- **Backend**: FastAPI with Python SQLAlchemy, Pydantic, and Celery task broker.

### Reasons for Porting to Spring Boot + HTMX:
1. **Unified Enterprise Control Plane**: Spring Boot 3 provides robust multi-threading, transactions, connection pooling, and unified security out of the box.
2. **Hypermedia-Driven UI**: Thymeleaf server-side rendering combined with HTMX provides dynamic single-page responsiveness without the memory and bundle footprint of client-side JavaScript frameworks.
3. **Resilient ML Execution**: A hybrid strategy where the Java backend orchestrates jobs, handles file parsing, and gracefully falls back to simulation mode when external Python compute clusters are offline.

---

## 2. Core Components

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

### Domain Entities (JPA / PostgreSQL / H2)
- `DatasetEntity`: Metadata, file paths, rows, columns, quality metrics, and sample preview data.
- `TrainingJobEntity`: Lifecycle state (`pending`, `running`, `completed`, `failed`), progress percentage, and hyperparameter configuration JSON.
- `ExperimentRunEntity`: Individual trial metrics and execution logs.
- `TrainedModelEntity`: Model architecture, accuracy, F1-score, ROC-AUC, RMSE, artifact file path, and SHAP attributions.

---

## 3. The Axiom Obsidian Visual System

The design system implements the specifications in `code/DESIGN.md`:
- **Surface**: `#0e0e10` (Dark canvas), `#19191d` (Elevated cards), `#25252b` (Variant)
- **Primary Accent**: `#d0bcff` (Soft violet), `#7c3aed` (Deep violet buttons)
- **Text**: `#e7e4ec` (Primary readable), `#acaab1` (Muted labels)
- **Status Indicators**: Emerald (`#34d399`) for healthy systems, amber (`#fbbf24`) for running experiments.
