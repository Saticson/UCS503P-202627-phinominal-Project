package com.axiomcloud.service;

import com.axiomcloud.dto.MarketplaceListingDto;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Service
public class MarketplaceService {

    private final Map<String, MarketplaceListingDto> listings = new ConcurrentHashMap<>();

    public MarketplaceService() {
        seedListings();
    }

    private void seedListings() {
        // Computer Vision Models
        addListing(new MarketplaceListingDto(
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
        ));

        addListing(new MarketplaceListingDto(
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
        ));

        addListing(new MarketplaceListingDto(
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
        ));

        addListing(new MarketplaceListingDto(
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
        ));

        // Tabular Models
        addListing(new MarketplaceListingDto(
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
        ));

        addListing(new MarketplaceListingDto(
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
        ));

        addListing(new MarketplaceListingDto(
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
        ));

        addListing(new MarketplaceListingDto(
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
        ));
    }

    private void addListing(MarketplaceListingDto listing) {
        listings.put(listing.getId(), listing);
    }

    public List<MarketplaceListingDto> listListings(String modality, String query, Boolean verifiedOnly) {
        return listings.values().stream()
                .filter(l -> {
                    if (modality != null && !modality.isBlank() && !modality.equalsIgnoreCase("ALL")) {
                        if (!l.getModality().equalsIgnoreCase(modality)) return false;
                    }
                    if (verifiedOnly != null && verifiedOnly) {
                        if (!l.isVerified()) return false;
                    }
                    if (query != null && !query.isBlank()) {
                        String q = query.toLowerCase();
                        boolean matchTitle = l.getTitle().toLowerCase().contains(q);
                        boolean matchDesc = l.getDescription().toLowerCase().contains(q);
                        boolean matchTag = l.getTags() != null && l.getTags().stream().anyMatch(t -> t.toLowerCase().contains(q));
                        if (!matchTitle && !matchDesc && !matchTag) return false;
                    }
                    return true;
                })
                .sorted(Comparator.comparingDouble(MarketplaceListingDto::getMetricScore).reversed())
                .collect(Collectors.toList());
    }

    public Optional<MarketplaceListingDto> getListing(String id) {
        return Optional.ofNullable(listings.get(id));
    }

    public List<MarketplaceListingDto> recommendListings(String modality, String taskType) {
        return listings.values().stream()
                .filter(l -> {
                    if (modality != null && !modality.isBlank() && !modality.equalsIgnoreCase("ALL")) {
                        if (!l.getModality().equalsIgnoreCase(modality)) return false;
                    }
                    if (taskType != null && !taskType.isBlank()) {
                        if (!l.getTaskType().equalsIgnoreCase(taskType)) return false;
                    }
                    return true;
                })
                .sorted(Comparator.comparingDouble(MarketplaceListingDto::getMetricScore).reversed())
                .limit(4)
                .collect(Collectors.toList());
    }

    public MarketplaceListingDto publish(MarketplaceListingDto dto) {
        if (dto.getId() == null || dto.getId().isBlank()) {
            dto.setId("custom-" + UUID.randomUUID().toString().substring(0, 8));
        }
        if (dto.getSellerName() == null || dto.getSellerName().isBlank()) {
            dto.setSellerName("Community Seller");
        }
        if (dto.getPrice() == null || dto.getPrice().isBlank()) {
            dto.setPrice("Free");
        }
        if (dto.getDownloadUrl() == null) {
            dto.setDownloadUrl("/api/marketplace/" + dto.getId() + "/download");
        }
        listings.put(dto.getId(), dto);
        return dto;
    }

    public int count() {
        return listings.size();
    }
}
