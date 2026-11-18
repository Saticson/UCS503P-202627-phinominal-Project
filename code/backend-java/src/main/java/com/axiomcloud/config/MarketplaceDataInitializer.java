package com.axiomcloud.config;

import com.axiomcloud.dto.MarketplaceListingDto;
import com.axiomcloud.model.MarketplaceListingEntity;
import com.axiomcloud.repository.MarketplaceListingRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class MarketplaceDataInitializer implements CommandLineRunner {

    private static final Logger logger = LoggerFactory.getLogger(MarketplaceDataInitializer.class);

    private final MarketplaceListingRepository repository;

    public MarketplaceDataInitializer(MarketplaceListingRepository repository) {
        this.repository = repository;
    }

    @Override
    public void run(String... args) {
        if (repository.count() > 0) {
            return;
        }

        logger.info("Seeding initial verified marketplace listings into database...");

        List<MarketplaceListingDto> initialListings = List.of(
                new MarketplaceListingDto(
                        "resnet-50-imagenet",
                        "ResNet-50 Enterprise Visual Classifier",
                        "High-performance deep convolutional network fine-tuned on 1,000 visual object categories with production ONNX export.",
                        "COMPUTER_VISION",
                        "CLASSIFICATION",
                        "PyTorch",
                        0.924,
                        "Top-1 Accuracy",
                        14,
                        "Free",
                        true,
                        "Sukio Core Labs",
                        "[3, 224, 224] Float32",
                        "Apache-2.0",
                        List.of("Vision", "ResNet", "ImageNet", "Edge-Ready")
                ),
                new MarketplaceListingDto(
                        "mobilenet-v3-digits",
                        "MobileNet-V3 Ultra-Fast Digit Classifier",
                        "Sub-5ms edge neural network engineered for real-time optical character and handwritten digit recognition.",
                        "COMPUTER_VISION",
                        "CLASSIFICATION",
                        "ONNX",
                        0.991,
                        "Accuracy",
                        3,
                        "Free",
                        true,
                        "Sukio Core Labs",
                        "[1, 28, 28] Grayscale",
                        "MIT",
                        List.of("Vision", "MobileNet", "Edge", "Low-Latency")
                ),
                new MarketplaceListingDto(
                        "yolov8-edge-defect",
                        "YOLOv8 Industrial Defect Inspector",
                        "Real-time surface and solder defect detection model with calibrated bounding box outputs for manufacturing assembly lines.",
                        "COMPUTER_VISION",
                        "DETECTION",
                        "PyTorch",
                        0.896,
                        "mAP@0.5",
                        22,
                        "$49",
                        true,
                        "Phinominal AI Systems",
                        "[3, 640, 640] RGB",
                        "Commercial",
                        List.of("Vision", "YOLO", "Inspection", "Manufacturing")
                ),
                new MarketplaceListingDto(
                        "vit-tiny-patch16",
                        "Vision Transformer (ViT-Tiny/16) Feature Extractor",
                        "Compact self-attention architecture providing high-dimensional visual embeddings for image retrieval and zero-shot categorization.",
                        "COMPUTER_VISION",
                        "CLASSIFICATION",
                        "PyTorch",
                        0.910,
                        "Top-1 Accuracy",
                        28,
                        "Free",
                        false,
                        "Community Contributor",
                        "[3, 224, 224] RGB",
                        "Apache-2.0",
                        List.of("Vision", "Transformer", "Embeddings")
                ),
                new MarketplaceListingDto(
                        "xgboost-fraud-guard",
                        "XGBoost Sub-Millisecond Fraud Detector",
                        "Gradient boosted decision tree ensemble optimized for high-throughput financial transaction anomaly detection.",
                        "TABULAR",
                        "CLASSIFICATION",
                        "Scikit-Learn",
                        0.985,
                        "AUROC",
                        1,
                        "Free",
                        true,
                        "Sukio Core Labs",
                        "Vector [1, 24] Tabular Features",
                        "MIT",
                        List.of("Tabular", "FinTech", "Fraud", "Gradient-Boosted")
                ),
                new MarketplaceListingDto(
                        "lightgbm-customer-churn",
                        "LightGBM Customer Lifetime & Churn Predictor",
                        "Scalable classification engine for recurring SaaS customer telemetry and retention scoring.",
                        "TABULAR",
                        "CLASSIFICATION",
                        "Scikit-Learn",
                        0.912,
                        "F1 Score",
                        2,
                        "$29",
                        true,
                        "Phinominal AI Systems",
                        "Vector [1, 18] Customer Metrics",
                        "Commercial",
                        List.of("Tabular", "Churn", "Enterprise", "LightGBM")
                ),
                new MarketplaceListingDto(
                        "catboost-real-estate-regressor",
                        "CatBoost Spatial Property Valuation Model",
                        "Gradient boosting pipeline with native categorical feature encoding for multi-region housing and price regressions.",
                        "TABULAR",
                        "REGRESSION",
                        "PyTorch",
                        0.941,
                        "R² Score",
                        4,
                        "Free",
                        true,
                        "Sukio Core Labs",
                        "Vector [1, 12] Spatial Features",
                        "Apache-2.0",
                        List.of("Tabular", "Regression", "Real-Estate", "CatBoost")
                ),
                new MarketplaceListingDto(
                        "survival-hazard-ensemble",
                        "Clinical Survival & Mortality Hazard Network",
                        "Cox proportional hazards neural baseline for time-to-event and clinical outcome risk stratification.",
                        "TABULAR",
                        "CLASSIFICATION",
                        "Scikit-Learn",
                        0.948,
                        "C-Index",
                        2,
                        "Free",
                        true,
                        "Bio-AI Consortium",
                        "Vector [1, 10] Biomarkers",
                        "MIT",
                        List.of("Tabular", "Healthcare", "Survival", "Clinical")
                )
        );

        for (MarketplaceListingDto dto : initialListings) {
            repository.save(MarketplaceListingEntity.fromDto(dto));
        }

        logger.info("Successfully seeded {} marketplace listings into database.", repository.count());
    }
}
