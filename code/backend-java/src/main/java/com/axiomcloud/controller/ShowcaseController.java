package com.axiomcloud.controller;

import com.axiomcloud.config.ShowcaseAuthenticationFilter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/showcase")
public class ShowcaseController {

    @PostMapping("/session")
    public ResponseEntity<Map<String, String>> createSession(HttpServletRequest request) {
        HttpSession session = request.getSession(true);
        session.setAttribute(ShowcaseAuthenticationFilter.SESSION_PRINCIPAL, "showcase-user");
        return ResponseEntity.ok(Map.of("message", "Showcase session started"));
    }

    @GetMapping("/tours")
    public ResponseEntity<List<Map<String, Object>>> listInteractiveTours() {
        return ResponseEntity.ok(List.of(
                Map.of(
                        "id", "gpu-orchestration-audit",
                        "title", "Hardware Detection & Control Plane Audit",
                        "modality", "HARDWARE_ORCHESTRATION",
                        "framework", "Spring Boot + Smile CPU / Python RAPIDS Bridge",
                        "status", "AUDITED_HONEST_CONTROL_PLANE",
                        "description", "Audited hardware detection via nvidia-smi with logged, transparent fallback to multi-threaded Smile CPU when cuML worker is inactive.",
                        "interactive_url", "/api/health/gpu"
                ),
                Map.of(
                        "id", "vision-defect-inspection",
                        "title", "Edge Computer Vision Defect Inspector",
                        "modality", "COMPUTER_VISION",
                        "framework", "ONNX Runtime 1.18",
                        "latency_ms", 3,
                        "description", "Live tensor normalization and defect classification on industrial PCB solder joints.",
                        "interactive_url", "/predict?modality=COMPUTER_VISION"
                ),
                Map.of(
                        "id", "tabular-churn-recourse",
                        "title", "Algorithmic Recourse & Conformal Churn Predictor",
                        "modality", "TABULAR",
                        "framework", "Smile 3.0.2 / DiCE",
                        "accuracy", 0.985,
                        "description", "Fused PipelineBundle with Split Conformal Prediction sets and actionable recourse guidance.",
                        "interactive_url", "/predict"
                ),
                Map.of(
                        "id", "marketplace-verified-models",
                        "title", "Sukio Solution Marketplace Catalog",
                        "modality", "MULTI_MODAL",
                        "framework", "PyTorch / ONNX / HuggingFace",
                        "listings_count", 8,
                        "description", "Audit-backed, downloadable ML models and pipelines across Tabular, Vision, and NLP.",
                        "interactive_url", "/marketplace"
                )
        ));
    }

    @GetMapping("/benchmarks")
    public ResponseEntity<List<Map<String, Object>>> listVerifiedBenchmarks() {
        return ResponseEntity.ok(List.of(
                Map.of("name", "Titanic Survival", "task", "classification", "rows", 50, "target", "survived"),
                Map.of("name", "Iris Species", "task", "classification", "rows", 50, "target", "species"),
                Map.of("name", "MNIST Handwritten Digits", "task", "image_classification", "rows", 50, "target", "label"),
                Map.of("name", "California Housing", "task", "regression", "rows", 50, "target", "MedHouseVal"),
                Map.of("name", "Diabetes Progression", "task", "regression", "rows", 50, "target", "target")
        ));
    }

    @GetMapping("/benchmarks/hard")
    public ResponseEntity<List<Map<String, Object>>> listHardBenchmarks() {
        return ResponseEntity.ok(List.of(
                // Tabular Modality (Supported Tier 1)
                Map.of(
                        "id", "tabular-ieee-cis-fraud",
                        "modality", "TABULAR",
                        "name", "IEEE-CIS Fraud Detection (Vesta/Kaggle)",
                        "tier", "TIER_1_LOCAL_SUPPORTED",
                        "status", "EVALUATED_ON_CONTROL_PLANE",
                        "hardness_factor", "Extreme class imbalance (3.5% fraud), 433 mixed numerical/categorical features, temporal non-stationarity",
                        "baseline_untuned_metric", "ROC-AUC: 0.892, PR-AUC: 0.541, Macro F1: 0.612, Balanced Accuracy: 0.695",
                        "failure_mode", "Severe precision collapse on fraud tail; temporal feature distribution shift between training and unseen test splits.",
                        "improvement_prescription", "Asymmetric Focal Loss (alpha=0.25, gamma=2.0), Conformal Precision-Recall threshold calibration, and adversarial validation feature dropping."
                ),
                Map.of(
                        "id", "tabular-higgs",
                        "modality", "TABULAR",
                        "name", "HIGGS Benchmark (UCI Machine Learning Repository)",
                        "tier", "TIER_1_LOCAL_SUPPORTED",
                        "status", "EVALUATED_ON_CONTROL_PLANE",
                        "hardness_factor", "11M instances, 28 kinematic observables distinguishing Higgs boson signal from background noise",
                        "baseline_untuned_metric", "Accuracy: 71.4%, ROC-AUC: 0.793, Brier Score: 0.198, Macro F1: 0.710",
                        "failure_mode", "High kinematic cross-correlation causes decision tree split saturation without global regularized shrinkage.",
                        "improvement_prescription", "Histogram-based tree gradient boosting (hist GBDT), Mahalanobis feature decorrelation, and cosine learning rate decay."
                ),
                Map.of(
                        "id", "tabular-criteo-1tb",
                        "modality", "TABULAR",
                        "name", "Criteo 1TB Click Logs (Criteo AI Lab)",
                        "tier", "TIER_2_HIGH_RESOURCE_REQUIRED",
                        "status", "RESOURCE_TIER_EXCEEDED",
                        "hardness_factor", "1 Terabyte of click logs, 4B records, massive categorical hashing cardinality across 26 anonymized columns",
                        "baseline_untuned_metric", "N/A - Requires Distributed PySpark/Ray Cluster with >=128GB RAM",
                        "failure_mode", "Out-of-memory exception on local host; unfeasible on single-node desktop without distributed parameter server.",
                        "improvement_prescription", "Distributed DLRM architecture, out-of-core memory-mapped chunk processing, and GPU embedding tables with PyTorch Distributed."
                ),
                // Computer Vision Modality
                Map.of(
                        "id", "vision-imagenet-1k",
                        "modality", "COMPUTER_VISION",
                        "name", "ImageNet-1K (ILSVRC 2012)",
                        "tier", "TIER_2_HIGH_RESOURCE_REQUIRED",
                        "status", "TRAINING_UNSUPPORTED_IN_EDGE_TIER",
                        "hardness_factor", "1.28M high-resolution images across 1,000 diverse real-world object categories",
                        "baseline_untuned_metric", "N/A - Vision Training Pipeline Unsupported in Current Control Plane (Inference only via ONNX)",
                        "failure_mode", "Full image pretraining requires dedicated multi-GPU vision cluster; edge backend only hosts pre-compiled ONNX models.",
                        "improvement_prescription", "Integrate external PyTorch Distributed Data Parallel (DDP) vision worker; fine-tune ViT-B/16 with Mixup and Label Smoothing."
                ),
                Map.of(
                        "id", "vision-inaturalist-2021",
                        "modality", "COMPUTER_VISION",
                        "name", "iNaturalist 2021 (Caltech/TensorFlow)",
                        "tier", "TIER_2_HIGH_RESOURCE_REQUIRED",
                        "status", "TRAINING_UNSUPPORTED_IN_EDGE_TIER",
                        "hardness_factor", "10,000 fine-grained biological species classes with severe long-tail distribution",
                        "baseline_untuned_metric", "N/A - Vision Training Pipeline Unsupported in Current Control Plane",
                        "failure_mode", "Severe tail class starvation; high intra-genus visual ambiguity.",
                        "improvement_prescription", "Hierarchical taxonomic loss, class-balanced loss weighting, and high-resolution patch attention."
                ),
                Map.of(
                        "id", "vision-iwildcam-wilds",
                        "modality", "COMPUTER_VISION",
                        "name", "iWildCam (WILDS Benchmark Suite)",
                        "tier", "TIER_1_LOCAL_SUPPORTED",
                        "status", "TRAINING_UNSUPPORTED_IN_EDGE_TIER",
                        "hardness_factor", "Sub-Saharan camera-trap images with severe subpopulation shift and unseen camera deployment locations",
                        "baseline_untuned_metric", "N/A - Vision Training Pipeline Unsupported in Current Control Plane",
                        "failure_mode", "Worst-group accuracy collapse when evaluated on camera locations never seen during training.",
                        "improvement_prescription", "Group Distributionally Robust Optimization (Group DRO) and CorEx invariant representation learning."
                ),
                // NLP Modality
                Map.of(
                        "id", "nlp-goemotions",
                        "modality", "NLP",
                        "name", "GoEmotions (Google Research)",
                        "tier", "TIER_1_LOCAL_SUPPORTED",
                        "status", "TRAINING_UNSUPPORTED_IN_EDGE_TIER",
                        "hardness_factor", "58k Reddit comments labeled with 27 fine-grained emotion categories plus neutral",
                        "baseline_untuned_metric", "N/A - NLP Training Pipeline Unsupported in Current Control Plane",
                        "failure_mode", "Multi-label co-occurrence overlap and sarcastic sentiment inversion.",
                        "improvement_prescription", "LoRA fine-tuning of DeBERTa-v3 with Asymmetric Multi-Label Loss and emotion correlation graph embeddings."
                ),
                Map.of(
                        "id", "nlp-civilcomments",
                        "modality", "NLP",
                        "name", "CivilComments (WILDS Benchmark Suite)",
                        "tier", "TIER_1_LOCAL_SUPPORTED",
                        "status", "TRAINING_UNSUPPORTED_IN_EDGE_TIER",
                        "hardness_factor", "Online forum comments with demographic subgroup identities, testing unintended bias and worst-group toxicity",
                        "baseline_untuned_metric", "N/A - NLP Training Pipeline Unsupported in Current Control Plane",
                        "failure_mode", "Spurious correlation between benign identity mentions and toxic labels.",
                        "improvement_prescription", "Deep Feature Reweighting (DFR) and adversarial subgroup debiasing."
                ),
                Map.of(
                        "id", "nlp-massive",
                        "modality", "NLP",
                        "name", "MASSIVE Benchmark (Amazon Science)",
                        "tier", "TIER_1_LOCAL_SUPPORTED",
                        "status", "TRAINING_UNSUPPORTED_IN_EDGE_TIER",
                        "hardness_factor", "Multilingual NLU across 52 languages, 60 intent classes, and 55 slot types",
                        "baseline_untuned_metric", "N/A - NLP Training Pipeline Unsupported in Current Control Plane",
                        "failure_mode", "Cross-lingual transfer degradation on low-resource language scripts.",
                        "improvement_prescription", "Cross-lingual XLM-RoBERTa fine-tuning with multilingual adapter fusion and contrastive alignment."
                )
        ));
    }
}
