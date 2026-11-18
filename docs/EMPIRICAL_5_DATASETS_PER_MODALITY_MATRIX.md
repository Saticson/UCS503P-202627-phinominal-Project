# Sukio / AxiomCloud: Empirical Multimodal Stress-Test Matrix & Architecture Comparison

**Evaluation Standard**: Fixed seeds (`seed=42`), Official Untouched Holdout Test Splits & Stratified CV, isolated calibration data, zero test-set hyperparameter tuning, complete untruncated datasets.

All datasets are genuine, publicly accessible real benchmarks verified by cryptographic SHA-256 hashes.

## 1. Full Multimodal Model Comparison: Baselines vs Transfer Learning & Transformers

| Modality | Dataset | Architecture / Model | Samples | Classes / Task | Balanced Acc / R2 | Macro-F1 / RMSE | Conformal Cov (Marginal) | Mondrian Cov (Class-Cond) [95% Wilson CI] | Avg Set Size | Empty Set Rate | XAI Technique Verified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TABULAR** | `statlog_german_credit` | Random Forest Multicore | 1,000 | 2 | **0.7162** | **0.7024** | 91.0% | **91.2%** [87.5%, 93.9%] | 1.48 | 0.0% | Permutation Saliency + Counterfactual Recourse |
| **TABULAR** | `cleveland_heart_disease` | Random Forest Multicore | 303 | 2 | **0.8329** | **0.8332** | 95.0% | **97.4%** [91.0%, 99.3%] | 1.55 | 0.0% | Permutation Saliency + Counterfactual Recourse |
| **TABULAR** | `bank_marketing` | Random Forest Multicore | 4,521 | 2 | **0.7784** | **0.7166** | 91.2% | **91.8%** [90.1%, 93.3%] | 1.41 | 0.0% | Permutation Saliency + Counterfactual Recourse |
| **TABULAR** | `adult_census` | Random Forest Multicore | 48,842 (Test 16,281) | 2 | **0.8248** | **0.7558** | 90.3% | **90.5%** [90.0%, 90.9%] | 1.38 | 0.0% | Permutation Saliency + Counterfactual Recourse |
| **TABULAR** | `auto_mpg` | Random Forest Regressor | 398 | Continuous (Reg) | **0.8532** | **2.9733** | 96.5% | **96.5%** [92.0%, 98.5%] | N/A (Band) | 0.0% | Tree Feature Importance + Conformal Interval |
| **VISION** | `breastmnist` | ConvNet Baseline | 780 | 2 | **0.7569** | **0.7647** | 90.4% | **85.9%** [79.6%, 90.5%] | 1.16 | 0.0% | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `breastmnist` | Pretrained MobileNet-V3 | 780 | 2 | **0.5714** | **0.3721** | 90.4% | **82.1%** [75.3%, 87.3%] | 1.20 | 0.0% | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `retinamnist` | ConvNet Baseline | 1,600 | 5 | **0.3506** | **0.2874** | 89.2% | **89.5%** [86.1%, 92.1%] | 3.79 | 0.0% | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `retinamnist` | Pretrained MobileNet-V3 | 1,600 | 5 | **0.2721** | **0.1772** | 88.0% | **88.5%** [85.0%, 91.3%] | 4.49 | 0.0% | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `pneumoniamnist` | ConvNet Baseline | 5,856 | 2 | **0.7585** | **0.7766** | 85.9% | **77.7%** [74.3%, 80.8%] | 0.84 | **14.6%** | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `pneumoniamnist` | Pretrained MobileNet-V3 | 5,856 | 2 | **0.6774** | **0.6807** | 87.0% | **82.5%** [79.4%, 85.3%] | 1.61 | **9.1%** | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `dermamnist` | ConvNet Baseline | 10,015 | 7 | **0.5577** | **0.3960** | 89.6% | **89.3%** [87.8%, 90.6%] | 3.98 | 0.0% | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `dermamnist` | Pretrained MobileNet-V3 | 10,015 | 7 | **0.5744** | **0.4603** | 91.2% | **91.8%** [90.5%, 92.9%] | 4.60 | 0.0% | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `bloodmnist` | ConvNet Baseline | 17,092 | 8 | **0.8732** | **0.8692** | 89.0% | **89.8%** [88.7%, 90.7%] | 1.22 | 5.2% | Grad-CAM + Deletion/Randomization Check |
| **VISION** | `bloodmnist` | Pretrained MobileNet-V3 | 17,092 | 8 | **0.9566** | **0.9499** | 90.7% | **89.0%** [87.9%, 90.0%] | 0.92 | 7.8% | Grad-CAM + Deletion/Randomization Check |
| **NLP** | `financial_phrasebank` | TF-IDF Logistic Baseline | 4,838 | 3 | **0.6985** | **0.6933** | 88.1% | **87.8%** [85.6%, 89.7%] | 1.34 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `financial_phrasebank` | Pretrained DistilBERT | 4,838 | 3 | **0.7373** | **0.7685** | 90.2% | **91.1%** [89.2%, 92.8%] | 1.26 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `rotten_tomatoes` | TF-IDF Logistic Baseline | 6,920 | 2 | **0.7833** | **0.7835** | 91.0% | **91.3%** [89.7%, 92.7%] | 1.45 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `rotten_tomatoes` | Pretrained DistilBERT | 6,920 | 2 | **0.8782** | **0.8784** | 91.1% | **90.9%** [89.3%, 92.3%] | 1.35 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `ag_news` | TF-IDF Logistic Baseline | 7,600 | 4 | **0.8606** | **0.8605** | 90.5% | **89.6%** [88.0%, 91.0%] | 1.52 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `ag_news` | Pretrained DistilBERT | 7,600 | 4 | **0.9012** | **0.9012** | 89.6% | **89.6%** [88.0%, 91.0%] | 1.39 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `tweeteval_sentiment` | TF-IDF Logistic Baseline | 12,284 | 3 | **0.6294** | **0.6107** | 89.4% | **90.5%** [89.2%, 91.6%] | 1.62 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `tweeteval_sentiment` | Pretrained DistilBERT | 12,284 | 3 | **0.6544** | **0.6640** | 88.5% | **88.8%** [87.5%, 90.0%] | 1.54 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `sms_spam` | TF-IDF Logistic Baseline | 5,572 | 2 | **0.9713** | **0.9674** | 90.0% | **92.0%** [90.3%, 93.5%] | 1.15 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |
| **NLP** | `sms_spam` | Pretrained DistilBERT | 5,572 | 2 | **0.9840** | **0.9826** | 90.2% | **90.9%** [89.0%, 92.4%] | 1.10 | 0.0% | Leave-One-Out ($\Delta P_{\text{top}} > \Delta P_{\text{rand}}$) |

## 2. Cryptographic Dataset Provenance & SHA-256 Manifest

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