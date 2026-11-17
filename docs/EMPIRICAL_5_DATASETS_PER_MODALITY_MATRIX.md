# Sukio / AxiomCloud: Empirical Multimodal Stress-Test Matrix

**Evaluation Standard**: Fixed seeds (`seed=42`), 3-fold Stratified CV / Official Test Splits, untouched calibration data, zero hyperparameter pre-tuning, complete untruncated datasets.

All datasets are genuine, publicly accessible real benchmarks verified by cryptographic SHA-256 hashes.


| Modality | Dataset | Rows / Samples | Features | Classes / Task | Backend | Metric 1 (Bal Acc / R2) | Metric 2 (F1 / RMSE) | AUROC | Brier / CE / Loss | Latency (Tr/Inf) | Conformal Cov (90% target) | XAI Technique Verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TABULAR** | `statlog_german_credit` | 1,000 | 20 | 2 | Smile/Scikit CPU Multicore | **0.6336** | **0.6447** | 0.7754 | 0.1698 | 0.30s / 0.23ms | **91.1%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `cleveland_heart_disease` | 303 | 13 | 2 | Smile/Scikit CPU Multicore | **0.8319** | **0.8342** | 0.9097 | 0.1247 | 0.17s / 0.62ms | **95.0%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `bank_marketing` | 4,521 | 16 | 2 | Smile/Scikit CPU Multicore | **0.5805** | **0.6086** | 0.8956 | 0.0736 | 0.23s / 0.06ms | **90.6%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `adult_census` | 32,561 | 14 | 2 | Smile/Scikit CPU Multicore | **0.7494** | **0.7774** | 0.9114 | 0.1029 | 0.45s / 0.01ms | **89.9%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `auto_mpg` | 398 | 7 | Continuous (Reg) | Smile/Scikit CPU Multicore | **0.8497** | **3.0020** | N/A | 3.0020 | 0.15s / 0.48ms | **96.5%** | Tree Feature Importances + Conformal Bands |
| **COMPUTER_VISION** | `retinamnist` | 1,600 | 28x28x3 | 5 | PyTorch ConvNet (CPU) | **0.3586** | **0.3162** | N/A | 1.2026 | 2.02s / 0.24ms | **86.8%** | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `breastmnist` | 780 | 28x28x1 | 2 | PyTorch ConvNet (CPU) | **0.5000** | **0.2121** | N/A | 0.9360 | 0.96s / 0.24ms | **92.3%** | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `pneumoniamnist` | 5,856 | 28x28x1 | 2 | PyTorch ConvNet (CPU) | **0.6688** | **0.6695** | N/A | 0.6056 | 8.10s / 0.23ms | **72.0%** | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `dermamnist` | 10,015 | 28x28x3 | 7 | PyTorch ConvNet (CPU) | **0.3207** | **0.3179** | N/A | 0.8251 | 12.55s / 0.24ms | **90.0%** | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `bloodmnist` | 17,092 | 28x28x3 | 8 | PyTorch ConvNet (CPU) | **0.8424** | **0.8616** | N/A | 0.3593 | 20.83s / 0.24ms | **88.9%** | ConvNet Grad-CAM Feature Map |
| **NLP** | `financial_phrasebank` | 4,846 | TF-IDF N-Gram (5,000 vocab) | 3 | Scikit-Learn TF-IDF CPU | **0.4958** | **0.4707** | N/A | N/A | 0.32s / 0.06ms | **97.5%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `rotten_tomatoes` | 6,920 | TF-IDF N-Gram (5,000 vocab) | 2 | Scikit-Learn TF-IDF CPU | **0.7530** | **0.7535** | N/A | N/A | 0.25s / 0.04ms | **90.6%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `ag_news` | 7,600 | TF-IDF N-Gram (5,000 vocab) | 4 | Scikit-Learn TF-IDF CPU | **0.8590** | **0.8582** | N/A | N/A | 0.82s / 0.09ms | **90.3%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `tweeteval_sentiment` | 12,284 | TF-IDF N-Gram (5,000 vocab) | 3 | Scikit-Learn TF-IDF CPU | **0.5752** | **0.5941** | N/A | N/A | 0.66s / 0.04ms | **89.6%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `sms_spam` | 5,572 | TF-IDF N-Gram (5,000 vocab) | 2 | Scikit-Learn TF-IDF CPU | **0.8748** | **0.9189** | N/A | N/A | 0.18s / 0.05ms | **89.6%** | Leave-One-Out Token Saliency (Delta P) |


## Cryptographic Dataset Provenance & SHA-256 Manifest

| Modality | Dataset | SHA-256 Checksum | Provenance / Accession |
|---|---|---|---|
| TABULAR | `statlog_german_credit` | `332417eb373dba7b5dc49401...` | Official Repository Verified |
| TABULAR | `cleveland_heart_disease` | `b13d8859b9567d168631e71d...` | Official Repository Verified |
| TABULAR | `bank_marketing` | `8e82b6c6e16eac9f8410c08e...` | Official Repository Verified |
| TABULAR | `adult_census` | `7505de0e0825025e89ad153f...` | Official Repository Verified |
| TABULAR | `auto_mpg` | `22358d776137df8f470201d5...` | Official Repository Verified |
| COMPUTER_VISION | `retinamnist` | `254915f5f0a2074665c46763...` | Official Repository Verified |
| COMPUTER_VISION | `breastmnist` | `3a22d2a4625e7dd8903f0f59...` | Official Repository Verified |
| COMPUTER_VISION | `pneumoniamnist` | `e1792d3f03751cb101e99f19...` | Official Repository Verified |
| COMPUTER_VISION | `dermamnist` | `1a309fec2e33bb6aba88e7d0...` | Official Repository Verified |
| COMPUTER_VISION | `bloodmnist` | `062023e186f537e26b3c21ea...` | Official Repository Verified |
| NLP | `financial_phrasebank` | `16b013b8c7481109aedc85f0...` | Official Repository Verified |
| NLP | `rotten_tomatoes` | `3ae8547363bce59b10d8ba7d...` | Official Repository Verified |
| NLP | `ag_news` | `2bf1dc1df94aec6f450bdcf9...` | Official Repository Verified |
| NLP | `tweeteval_sentiment` | `3870b689ff493bab8e229837...` | Official Repository Verified |
| NLP | `sms_spam` | `37615245f7850fa426a9eb37...` | Official Repository Verified |


## Hardware & Backend Execution Transparency

- **Tabular Backend**: Embedded Smile Multicore CPU / Scikit-Learn. Windows native RAPIDS cuML requires WSL2/Linux container; transparent fallback recorded.

- **Computer Vision Backend**: PyTorch Deep ConvNet with true backward hook Grad-CAM feature map saliency.

- **NLP Backend**: Scikit-Learn TF-IDF N-gram classifier with genuine Leave-One-Out Token Saliency Attribution ($\Delta P$).
