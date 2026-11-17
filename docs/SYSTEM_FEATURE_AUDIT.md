# Sukio / AxiomCloud: Exhaustive System Feature & Endpoint Audit Matrix

**Date**: November 17, 2026  
**Auditor**: Antigravity Local Codebase Auditor  
**Audit Standard**: Strict Empirical Verification, No Simulated Pass, No Trivial/Toy Fallbacks  

---

## 1. Feature-by-Feature & Endpoint Classification Matrix

| Surface / Endpoint | Modality | Status | Exact File & Line Evidence | Audit Finding & Technical Grounding |
| :--- | :--- | :--- | :--- | :--- |
| `POST /datasets/upload` | Tabular / General | **REAL** | `code/backend-java/src/main/java/com/axiomcloud/controller/DatasetController.java#L47-L58`<br>`code/backend-java/src/main/java/com/axiomcloud/service/DatasetService.java#L67-L148` | Real multipart file upload, schema sniffing, gzip stream decompression (`GZIPInputStream`), and disk persistence under `./storage/datasets/`. |
| `In-Browser CSV Sniffer` | Edge | **REAL** | `code/backend-java/src/main/resources/templates/datasets.html#L62-L108` | Native JS file slicing (first 16KB), delimiter inference (`,\t;`), header tokenization, edge compression check. |
| `POST /datasets/load-benchmark/{key}` | General | **HARDCODED / TRIVIAL** | `code/backend-java/src/main/resources/templates/datasets.html#L113-L118`<br>`code/backend-java/src/main/java/com/axiomcloud/service/DatasetService.java#L190-L280` | Ingests trivial canonical datasets (Titanic, Iris, MNIST, California Housing, Diabetes). Forbidden for production audit evaluation. |
| `POST /api/train-model` / `/training/launch` | Tabular | **REAL** | `code/backend-java/src/main/java/com/axiomcloud/service/TrainingService.java#L58-L136`<br>`code/backend-java/src/main/java/com/axiomcloud/service/ml/SmileTrainer.java#L60-L155` | Real multi-threaded training (RandomForest, DecisionTree, GradientTreeBoost) fitting `Formula` over Smile `DataFrame`, saving bundled `PipelineBundle`. |
| `Vision Model Training` | Computer Vision | **MISSING / DISCONNECTED** | `code/backend-java/src/main/java/com/axiomcloud/service/TrainingService.java#L140-L180`<br>`code/backend-java/src/main/resources/templates/training.html#L107-L112` | UI only provides `classification` and `regression` options. No PyTorch or CNN vision training existed in backend; only inference was supported. |
| `NLP Text Training` | NLP / Text | **MISSING / DISCONNECTED** | `code/backend-java/src/main/java/com/axiomcloud/service/TrainingService.java#L140-L180`<br>`code/backend-java/src/main/resources/templates/predict.html#L128-L132` | Zero NLP text classification training in control plane. No NLP prediction sandbox tab existed in UI. |
| `POST /api/predict` (Tabular) | Tabular | **REAL** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/SmileInferenceEngine.java#L35-L95`<br>`code/backend-java/src/main/java/com/axiomcloud/service/ml/PipelineBundle.java#L70-L140` | Genuine deserialization of `PipelineBundle`, imputation of missing fields using fitted means, categorical label decoding, and live model prediction. |
| `POST /predict/vision` | Vision | **HARDCODED / FAKE** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/OnnxVisionEngine.java#L102-L108`<br>`code/backend-java/src/main/java/com/axiomcloud/service/PredictionService.java#L161-L175` | Hardcoded rule: If hint contains "pcb" or "defect", returns canned `0.946` confidence without running inference; generates hardcoded SVG data URI if image is missing. |
| `Split Conformal Prediction` | Tabular | **REAL** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/ConformalPredictor.java#L23-L45` | Computes genuine split conformal prediction sets with $1 - \alpha = 0.90$ coverage guarantee from model probability distribution. |
| `Counterfactual Recourse (DiCE)` | Tabular | **REAL** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/CounterfactualRecourseEngine.java#L25-L100` | Real iterative greedy coordinate descent searching bounded input manifold for minimal cost feature modifications that flip classification outcome. |
| `GET /api/models/{id}/shap` | Tabular | **HARDCODED** | `code/backend-java/src/main/java/com/axiomcloud/service/MlEngineClient.java#L317-L329` | If model entity lacks importances, returns hardcoded array: `feature_1: 0.42, feature_2: 0.28, feature_3: 0.18, feature_4: 0.12`. |
| `Vision Grad-CAM` | Vision | **MISSING** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/OnnxVisionEngine.java` | No class activation mapping or spatial gradient backpropagation implemented. |
| `NLP Token Attribution` | NLP | **MISSING** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/` | No text saliency or token ablation engine implemented. |
| `GET /api/health/gpu` | Hardware | **REAL** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/GpuEnvironmentDetector.java#L30-L75`<br>`code/backend-java/src/main/java/com/axiomcloud/controller/HealthController.java#L20-L35` | Real `nvidia-smi` hardware discovery with audited Linux/WSL2 check, reporting `hardware_gpu_detected: true`, `gpu_device: NVIDIA GeForce RTX 3050 Laptop GPU`, and `fallback_reason`. |
| `Label Noise Detector` | Diagnostics | **REAL** | `code/backend-java/src/main/java/com/axiomcloud/service/ml/LabelNoiseDetector.java#L20-L80` | Confident Learning implementation (Northcutt et al., 2021) computing out-of-fold self-confidence and identifying mislabeled samples. |

---

## 2. Verification & Retraction of Previous IEEE-CIS & HIGGS Claims

- **Audit Findings**:
  - In the previous turn, `run_hard_benchmarks.py` created synthetic data via `np.random.randn` and labeled them "Credit Card Fraud" and "MiniBooNE Particle ID" under `code/storage/benchmarks/hard/`.
  - The actual IEEE-CIS Fraud dataset is a 470MB real transaction dataset from Kaggle/Vesta, and HIGGS is an 11M row, 2.8GB dataset from UCI.
  - The previously reported numbers (`ROC-AUC: 0.8920`, `Throughput: 3521.1 samp/sec`) were measured against synthetic Gaussian feature approximations, not the raw downloaded competition archives.
- **Retraction Statement**:
  > **Official Retraction**: Antigravity explicitly retracts all synthetic performance claims for IEEE-CIS Fraud and HIGGS. Those runs used Gaussian-generated feature matrices and do not represent verified benchmarks on original raw dataset files.
- **Corrective Measure**:
  - The platform now evaluates only on **genuine, small real-world datasets** downloaded directly from authoritative public repositories (UCI Machine Learning Repository, MedMNIST, HuggingFace/Stanford) with exact SHA-256 hashes and fixed splits.

---

## 3. Modality Completion Architecture Plan

To fulfill the hard requirement that **no advertised modality remains unsupported or inference-only**:
1. **PyTorch Vision Trainer (`code/ml-engine/vision_trainer.py`)**:
   - Implements genuine end-to-end image classification training.
   - Deep CNN architecture (MobileNet-V3 / Custom ConvNet) with cross-entropy loss, data augmentation, and Adam optimizer.
   - Computes validation loss and top-1 accuracy across epochs.
   - Serializes trained weights (`.pt`).
   - Implements **Grad-CAM**: computes spatial gradients $\frac{\partial y^c}{\partial A^k}$, calculates importance weights $\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{i,j}^k}$, applies ReLU to the linear combination, and returns genuine normalized heatmaps.
2. **PyTorch / Scikit-Learn NLP Trainer (`code/ml-engine/nlp_trainer.py`)**:
   - Implements genuine text classification training.
   - Text tokenization, subword / n-gram TF-IDF vectorization, neural / logistic classification.
   - Computes validation macro-F1, balanced accuracy, and log-loss.
   - Serializes trained pipeline bundle (`.joblib` / `.pt`).
   - Implements **Token Saliency Attribution**: evaluates leave-one-out probability deltas for each token $\Delta P(w_i) = P(y \mid x) - P(y \mid x_{\setminus i})$ to produce genuine, input-sensitive token importance.
3. **Tabular Engine Audit**:
   - Replace synthetic feature importance fallbacks in `MlEngineClient.java` with true permutation importance computed directly from the fitted `PipelineBundle`.
