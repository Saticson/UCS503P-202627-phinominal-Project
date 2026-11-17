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
}
