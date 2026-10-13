package com.axiomcloud.service;

import com.axiomcloud.dto.ExpertCommissionDto;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Service
public class CommissionService {

    private final Map<String, ExpertCommissionDto> commissions = new ConcurrentHashMap<>();

    public CommissionService() {
        seedCommissions();
    }

    private void seedCommissions() {
        add(new ExpertCommissionDto(
                "req-101",
                "Apex Robotics Inc.",
                "lead@apexrobotics.io",
                "High-Speed PCB Solder Defect Detector with Grad-CAM Visual Heatmaps",
                "COMPUTER_VISION",
                "DETECTION",
                "$1,200",
                "Need an ultra-low latency model fine-tuned on microscope camera captures (1080p, 60fps) with visual saliency/Grad-CAM overlays for QA operators.",
                "OPEN"
        ));

        add(new ExpertCommissionDto(
                "req-102",
                "FinSure Credit Labs",
                "risk@finsure.co",
                "Constrained Tabular Fraud Classifier with Strict Monotonicity Guarantees",
                "TABULAR",
                "CLASSIFICATION",
                "$850",
                "Require an XGBoost/LightGBM model adhering to Basel regulatory fairness constraints, monotonic debt-to-income splits, and sub-millisecond inference.",
                "MATCHED"
        ));
    }

    public List<ExpertCommissionDto> listCommissions() {
        return commissions.values().stream()
                .sorted(Comparator.comparing(ExpertCommissionDto::getCreatedAt).reversed())
                .collect(Collectors.toList());
    }

    public ExpertCommissionDto add(ExpertCommissionDto dto) {
        if (dto.getId() == null || dto.getId().isBlank()) {
            dto.setId("req-" + UUID.randomUUID().toString().substring(0, 6));
        }
        commissions.put(dto.getId(), dto);
        return dto;
    }

    public int count() {
        return commissions.size();
    }
}
