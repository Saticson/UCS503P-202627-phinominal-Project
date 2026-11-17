package com.axiomcloud.service;

import com.axiomcloud.dto.ExpertCommissionDto;
import com.axiomcloud.model.ExpertCommissionEntity;
import com.axiomcloud.repository.ExpertCommissionRepository;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class CommissionService {

    private final ExpertCommissionRepository commissionRepository;

    public CommissionService(ExpertCommissionRepository commissionRepository) {
        this.commissionRepository = commissionRepository;
    }

    @PostConstruct
    public void seedInitialCommissions() {
        if (commissionRepository.count() == 0) {
            commissionRepository.save(new ExpertCommissionEntity(
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

            commissionRepository.save(new ExpertCommissionEntity(
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
    }

    public List<ExpertCommissionDto> listCommissions() {
        return commissionRepository.findAllByOrderByCreatedAtDesc().stream()
                .map(ExpertCommissionEntity::toDto)
                .collect(Collectors.toList());
    }

    public Optional<ExpertCommissionDto> getCommission(String id) {
        return commissionRepository.findById(id).map(ExpertCommissionEntity::toDto);
    }

    public ExpertCommissionDto add(ExpertCommissionDto dto) {
        if (dto.getId() == null || dto.getId().isBlank()) {
            dto.setId("req-" + UUID.randomUUID().toString().substring(0, 6));
        }
        ExpertCommissionEntity entity = ExpertCommissionEntity.fromDto(dto);
        ExpertCommissionEntity saved = commissionRepository.save(entity);
        return saved.toDto();
    }

    public int count() {
        return (int) commissionRepository.count();
    }
}
