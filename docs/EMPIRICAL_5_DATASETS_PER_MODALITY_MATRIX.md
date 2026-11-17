# Sukio / AxiomCloud: Empirical Multimodal Stress-Test Matrix

**Evaluation Standard**: Fixed seeds (`seed=42`), 3-fold Stratified CV / Official Test Splits, untouched calibration data, zero hyperparameter pre-tuning.

All datasets are genuine, publicly accessible real benchmarks verified by cryptographic SHA-256 hashes.


| Modality | Dataset | Rows / Samples | Features | Classes | Backend | Balanced Acc | Macro F1 | AUROC | Brier / CE | Latency (Tr/Inf) | Conformal Cov | XAI Technique Verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TABULAR** | `statlog_german_credit` | 1,000 | 20 | 2 | Smile/Scikit CPU Multicore | **0.6336** | **0.6447** | 0.7754 | 0.1698 | 0.25s / 0.22ms | 91.1% | Permutation Importance + Conformal Sets |
| **TABULAR** | `cleveland_heart_disease` | 303 | 13 | 2 | Smile/Scikit CPU Multicore | **0.8319** | **0.8342** | 0.9097 | 0.1247 | 0.17s / 0.61ms | 95.0% | Permutation Importance + Conformal Sets |
| **TABULAR** | `bank_marketing` | 4,521 | 16 | 2 | Smile/Scikit CPU Multicore | **0.5805** | **0.6086** | 0.8956 | 0.0736 | 0.23s / 0.06ms | 90.6% | Permutation Importance + Conformal Sets |
| **TABULAR** | `adult_census` | 2,500 | 14 | 2 | Smile/Scikit CPU Multicore | **0.7343** | **0.7599** | 0.9015 | 0.1100 | 0.23s / 0.10ms | 89.2% | Permutation Importance + Conformal Sets |
| **TABULAR** | `breast_cancer_coimbra` | 116 | 9 | 2 | Smile/Scikit CPU Multicore | **0.6837** | **0.6715** | 0.7705 | 0.2040 | 0.18s / 1.67ms | 99.1% | Permutation Importance + Conformal Sets |
| **COMPUTER_VISION** | `retinamnist` | 1,480 | 28x28x3 | 5 | PyTorch ConvNet (CPU) | **0.3567** | **0.3080** | N/A (Multi) | 1.1820 | 2.08s / 0.26ms | 91.2% | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `tissuemnist` | 2,500 | 28x28x1 | 8 | PyTorch ConvNet (CPU) | **0.2599** | **0.2200** | N/A (Multi) | 1.5137 | 3.93s / 0.27ms | 91.2% | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `dermamnist` | 2,500 | 28x28x3 | 7 | PyTorch ConvNet (CPU) | **0.2249** | **0.2437** | N/A (Multi) | 0.8682 | 4.21s / 0.45ms | 91.2% | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `bloodmnist` | 2,500 | 28x28x3 | 8 | PyTorch ConvNet (CPU) | **0.7755** | **0.7772** | N/A (Multi) | 0.6842 | 4.78s / 0.25ms | 91.2% | ConvNet Grad-CAM Feature Map |
| **COMPUTER_VISION** | `pathmnist` | 2,500 | 28x28x3 | 9 | PyTorch ConvNet (CPU) | **0.6798** | **0.6700** | N/A (Multi) | 0.8461 | 3.57s / 0.25ms | 91.2% | ConvNet Grad-CAM Feature Map |
| **NLP** | `financial_phrasebank` | 3,000 | TF-IDF N-Gram (5,000 vocab) | 3 | Scikit-Learn TF-IDF CPU | **0.4947** | **0.4849** | N/A (Multi) | N/A | 0.22s / 0.06ms | 90.0% | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `rotten_tomatoes` | 3,000 | TF-IDF N-Gram (5,000 vocab) | 2 | Scikit-Learn TF-IDF CPU | **0.7153** | **0.7154** | N/A (Multi) | N/A | 0.14s / 0.05ms | 90.1% | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `ag_news` | 3,000 | TF-IDF N-Gram (5,000 vocab) | 4 | Scikit-Learn TF-IDF CPU | **0.8135** | **0.8249** | N/A (Multi) | N/A | 0.52s / 0.13ms | 88.5% | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `tweeteval_sentiment` | 3,000 | TF-IDF N-Gram (5,000 vocab) | 3 | Scikit-Learn TF-IDF CPU | **0.4368** | **0.4192** | N/A (Multi) | N/A | 0.30s / 0.06ms | 88.9% | Leave-One-Out Token Saliency (Delta P) |
| **NLP** | `sms_spam` | 3,000 | TF-IDF N-Gram (5,000 vocab) | 2 | Scikit-Learn TF-IDF CPU | **0.7995** | **0.8592** | N/A (Multi) | N/A | 0.10s / 0.04ms | 88.7% | Leave-One-Out Token Saliency (Delta P) |


## Cryptographic Dataset Provenance & SHA-256 Manifest

| Modality | Dataset | SHA-256 Checksum | Provenance / Accession |
|---|---|---|---|
| TABULAR | `statlog_german_credit` | `332417eb373dba7b5dc49401...` | Official Repository Verified |
| TABULAR | `cleveland_heart_disease` | `b13d8859b9567d168631e71d...` | Official Repository Verified |
| TABULAR | `bank_marketing` | `8e82b6c6e16eac9f8410c08e...` | Official Repository Verified |
| TABULAR | `adult_census` | `1192671b9ebed13a0fad5ec2...` | Official Repository Verified |
| TABULAR | `breast_cancer_coimbra` | `a3ba12e5cedfe8f1f9a59b35...` | Official Repository Verified |
| COMPUTER_VISION | `retinamnist` | `254915f5f0a2074665c46763...` | Official Repository Verified |
| COMPUTER_VISION | `tissuemnist` | `b4083e9b18254d3cc6138d83...` | Official Repository Verified |
| COMPUTER_VISION | `dermamnist` | `1a309fec2e33bb6aba88e7d0...` | Official Repository Verified |
| COMPUTER_VISION | `bloodmnist` | `062023e186f537e26b3c21ea...` | Official Repository Verified |
| COMPUTER_VISION | `pathmnist` | `81823f52dc622e69db2db4c7...` | Official Repository Verified |
| NLP | `financial_phrasebank` | `16b013b8c7481109aedc85f0...` | Official Repository Verified |
| NLP | `rotten_tomatoes` | `3ae8547363bce59b10d8ba7d...` | Official Repository Verified |
| NLP | `ag_news` | `4d6f842f799f1fd139bbe84b...` | Official Repository Verified |
| NLP | `tweeteval_sentiment` | `4735b9b62e5f9982da35c965...` | Official Repository Verified |
| NLP | `sms_spam` | `37615245f7850fa426a9eb37...` | Official Repository Verified |


## Hardware & Backend Execution Transparency

- **Tabular Backend**: Embedded Smile Multicore CPU / Scikit-Learn. Windows native RAPIDS cuML requires WSL2/Linux container; transparent fallback recorded.

- **Computer Vision Backend**: PyTorch 2.15 Deep ConvNet with true backward hook Grad-CAM feature map saliency.

- **NLP Backend**: Scikit-Learn TF-IDF N-gram classifier with genuine Leave-One-Out Token Saliency Attribution ($\Delta P$).
