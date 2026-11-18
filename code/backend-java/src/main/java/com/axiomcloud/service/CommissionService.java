package com.axiomcloud.service;

import com.axiomcloud.dto.ExpertCommissionDto;
import com.axiomcloud.model.CommissionProposalEntity;
import com.axiomcloud.model.ExpertCommissionEntity;
import com.axiomcloud.repository.CommissionProposalRepository;
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
    private final CommissionProposalRepository proposalRepository;

    public CommissionService(ExpertCommissionRepository commissionRepository,
                             CommissionProposalRepository proposalRepository) {
        this.commissionRepository = commissionRepository;
        this.proposalRepository = proposalRepository;
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

            // Seed a sample proposal for req-102
            proposalRepository.save(new CommissionProposalEntity(
                    "prop-201",
                    "req-102",
                    "Dr. Elena Rostova",
                    "elena@quantml.tech",
                    "$800",
                    3,
                    "I will implement an exact monotonic Gradient Boosted Decision Tree with sub-millisecond C-bindings and comprehensive Basel regulatory reports."
            ));
        }
    }

    public List<ExpertCommissionDto> listCommissions() {
        return commissionRepository.findAllByOrderByCreatedAtDesc().stream()
                .map(entity -> {
                    ExpertCommissionDto dto = entity.toDto();
                    List<CommissionProposalEntity> proposals = proposalRepository.findByCommissionIdOrderByCreatedAtDesc(dto.getId());
                    dto.setProposals(proposals);
                    dto.setProposalCount(proposals.size());
                    return dto;
                })
                .collect(Collectors.toList());
    }

    public Optional<ExpertCommissionDto> getCommission(String id) {
        return commissionRepository.findById(id).map(entity -> {
            ExpertCommissionDto dto = entity.toDto();
            List<CommissionProposalEntity> proposals = proposalRepository.findByCommissionIdOrderByCreatedAtDesc(dto.getId());
            dto.setProposals(proposals);
            dto.setProposalCount(proposals.size());
            return dto;
        });
    }

    public ExpertCommissionDto add(ExpertCommissionDto dto) {
        if (dto.getId() == null || dto.getId().isBlank()) {
            dto.setId("req-" + UUID.randomUUID().toString().substring(0, 6));
        }
        ExpertCommissionEntity entity = ExpertCommissionEntity.fromDto(dto);
        ExpertCommissionEntity saved = commissionRepository.save(entity);
        return saved.toDto();
    }

    public CommissionProposalEntity addProposal(String commissionId, String expertName,
                                                String expertEmail, String bidAmount,
                                                int estimatedDays, String proposalText) {
        String propId = "prop-" + UUID.randomUUID().toString().substring(0, 6);
        CommissionProposalEntity entity = new CommissionProposalEntity(
                propId, commissionId, expertName, expertEmail, bidAmount, estimatedDays, proposalText
        );
        CommissionProposalEntity saved = proposalRepository.save(entity);

        // If commission was OPEN, update status to IN_REVIEW or MATCHED if applicable
        commissionRepository.findById(commissionId).ifPresent(comm -> {
            if ("OPEN".equals(comm.getStatus())) {
                comm.setStatus("MATCHED");
                commissionRepository.save(comm);
            }
        });

        return saved;
    }

    public List<CommissionProposalEntity> listProposals(String commissionId) {
        return proposalRepository.findByCommissionIdOrderByCreatedAtDesc(commissionId);
    }

    public int count() {
        return (int) commissionRepository.count();
    }
}
