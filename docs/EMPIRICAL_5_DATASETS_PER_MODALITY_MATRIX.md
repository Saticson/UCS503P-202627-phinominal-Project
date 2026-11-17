# Sukio / AxiomCloud: Empirical Multimodal Stress-Test Matrix

**Evaluation Standard**: Fixed seeds (`seed=42`), Official Train/Test Splits & 3-fold Stratified CV, untouched calibration data, zero hyperparameter pre-tuning, complete untruncated datasets.

All datasets are genuine, publicly accessible real benchmarks verified by cryptographic SHA-256 hashes.


| Modality | Dataset | Rows / Samples | Features | Classes / Task | Backend | Metric 1 (Bal Acc / R2) | Metric 2 (F1 / RMSE) | AUROC | Loss / Brier | Latency (Tr/Inf) | Conformal Cov (Marginal) | Mondrian Cov (Class-Cond) | XAI Technique Verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TABULAR** | `statlog_german_credit` | 1,000 | 20 | 2 | Smile/Scikit CPU Multicore | **0.6891** | **0.6851** | 0.7775 | 0.1780 | 0.17s / 0.20ms | 91.0% | **91.2%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `cleveland_heart_disease` | 303 | 13 | 2 | Smile/Scikit CPU Multicore | **0.8240** | **0.8250** | 0.9090 | 0.1243 | 0.14s / 0.55ms | 95.0% | **97.4%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `bank_marketing` | 4,521 | 16 | 2 | Smile/Scikit CPU Multicore | **0.7650** | **0.7106** | 0.8843 | 0.1042 | 0.18s / 0.04ms | 91.2% | **91.8%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `adult_census` | 48,842 | 14 | 2 | Smile/Scikit CPU Multicore | **0.8237** | **0.7539** | 0.9083 | 0.1324 | 0.50s / 0.01ms | 90.3% | **90.5%** | Permutation Importance + Conformal Sets |
| **TABULAR** | `auto_mpg` | 398 | 7 | Continuous (Reg) | Smile/Scikit CPU Multicore | **0.8497** | **3.0020** | N/A | 3.0020 | 0.11s / 0.44ms | 96.5% | **96.5%** | Tree Feature Importances + Conformal Bands |
| **COMPUTER_VISION** | `retinamnist` | 1,600 | 28x28x3 | 5 | PyTorch ConvNet (CPU) | **0.3928** | **0.3856** | N/A | 1.2661 | 5.99s / 0.20ms | 90.0% | **86.8%** | ConvNet Grad-CAM Feature Map (Faithfulness Verified) |
| **COMPUTER_VISION** | `breastmnist` | 780 | 28x28x1 | 2 | PyTorch ConvNet (CPU) | **0.7882** | **0.7882** | N/A | 0.4114 | 3.02s / 0.22ms | 91.7% | **87.2%** | ConvNet Grad-CAM Feature Map (Faithfulness Verified) |
| **COMPUTER_VISION** | `pneumoniamnist` | 5,856 | 28x28x1 | 2 | PyTorch ConvNet (CPU) | **0.8714** | **0.8868** | N/A | 0.3031 | 26.44s / 0.23ms | 80.4% | **79.2%** | ConvNet Grad-CAM Feature Map (Faithfulness Verified) |
| **COMPUTER_VISION** | `dermamnist` | 10,015 | 28x28x3 | 7 | PyTorch ConvNet (CPU) | **0.5500** | **0.3825** | N/A | 1.2056 | 38.30s / 0.21ms | 89.2% | **89.7%** | ConvNet Grad-CAM Feature Map (Faithfulness Verified) |
| **COMPUTER_VISION** | `bloodmnist` | 17,092 | 28x28x3 | 8 | PyTorch ConvNet (CPU) | **0.9245** | **0.9129** | N/A | 0.2144 | 66.88s / 0.21ms | 90.6% | **89.9%** | ConvNet Grad-CAM Feature Map (Faithfulness Verified) |
| **NLP** | `financial_phrasebank` | 4,838 | TF-IDF N-Gram (5,000 vocab) | 3 | Scikit-Learn TF-IDF CPU | **0.6985** | **0.6933** | N/A | N/A | 0.27s / 0.06ms | 88.1% | **87.8%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `rotten_tomatoes` | 6,920 | TF-IDF N-Gram (5,000 vocab) | 2 | Scikit-Learn TF-IDF CPU | **0.7574** | **0.7573** | N/A | N/A | 0.20s / 0.03ms | 90.9% | **91.0%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `ag_news` | 7,600 | TF-IDF N-Gram (5,000 vocab) | 4 | Scikit-Learn TF-IDF CPU | **0.8590** | **0.8582** | N/A | N/A | 0.68s / 0.08ms | 90.5% | **90.9%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `tweeteval_sentiment` | 12,284 | TF-IDF N-Gram (5,000 vocab) | 3 | Scikit-Learn TF-IDF CPU | **0.6250** | **0.6059** | N/A | N/A | 0.54s / 0.03ms | 89.4% | **89.5%** | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `sms_spam` | 5,572 | TF-IDF N-Gram (5,000 vocab) | 2 | Scikit-Learn TF-IDF CPU | **0.9639** | **0.9605** | N/A | N/A | 0.14s / 0.03ms | 90.0% | **89.8%** | Leave-One-Out Token Saliency (Delta P) |


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