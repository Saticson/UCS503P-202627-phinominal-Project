# Playwright Automated Testing Guide

Sukio includes a demo-level, presentation-ready Playwright end-to-end testing suite located under `code/tests/playwright/`.

---

## 1. Test Architecture

The testing framework runs headless Firefox / Chromium tests against the Spring Boot control plane at `http://localhost:8000`:

```
code/tests/playwright/
â”œâ”€â”€ global-setup.ts                  Directory verification & service health probes
â”œâ”€â”€ playwright.config.ts             Headless browser & viewport settings
â”œâ”€â”€ demo-e2e-journey.spec.ts         Complete multi-stage user journey
â”œâ”€â”€ demo-datasets.spec.ts            Dataset upload, benchmarks & diagnostics
â”œâ”€â”€ demo-training.spec.ts            AutoML model checklists & Optuna sweeps
â”œâ”€â”€ demo-models.spec.ts              Leaderboard rankings & SHAP attributions
â”œâ”€â”€ demo-predict.spec.ts             Live inference sandbox & confidence gauge
â”œâ”€â”€ demo-visual-capture.spec.ts      Presentation-grade artifact screenshots
â””â”€â”€ test-*.ts                        Underlying component & smoke tests
```

---

## 2. Running the Demo Suite

### Prerequisites
1. Ensure the Spring Boot backend is active:
   ```bash
   cd code/backend-java
   ./mvnw spring-boot:run -Dspring-boot.run.profiles=local
   ```
2. Navigate to the test suite directory:
   ```bash
   cd code/tests/playwright
   npm install
   ```

### Execution Commands
- **Run all demo journeys**:
  ```bash
  npm run test:demo
  ```
- **Run full test suite**:
  ```bash
  npm test
  ```
- **Capture demo screenshot artifacts**:
  ```bash
  npx playwright test demo-visual-capture
  ```
  Artifacts will be deposited in `code/tests/playwright/test-output/demo-artifacts/`.

---

## 3. Viewing Visual Test Artifacts (PNGs, GIFs, and Videos)

All test artifacts (screenshots, GIFs, video recordings, and reports) are kept **completely untracked in git** via `code/.gitignore`.

To inspect them visually:
- **Interactive Visual Gallery**:
  Open [`gallery.html`](file:///C:/Users/keshav/Documents/soft_desgn_proj/UCS503P-202627-phinominal-Project/code/tests/playwright/test-output/gallery.html) in any browser (or run `npm run gallery`). It displays an Axiom Obsidian dark-themed gallery with thumbnails and previews of all generated PNG screenshots and video recordings.
- **Playwright Trace & HTML Report**:
  ```bash
  npm run report
  ```
  Launches the official Playwright interactive report showing step-by-step DOM snapshots and network logs.
