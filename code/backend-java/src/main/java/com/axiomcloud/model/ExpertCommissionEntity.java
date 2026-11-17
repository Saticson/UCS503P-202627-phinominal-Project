package com.axiomcloud.model;

import com.axiomcloud.dto.ExpertCommissionDto;
import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "expert_commissions")
public class ExpertCommissionEntity {

    @Id
    @Column(nullable = false, length = 100)
    private String id;

    @Column(name = "client_name", nullable = false, length = 255)
    private String clientName;

    @Column(nullable = false, length = 255)
    private String email;

    @Column(nullable = false, length = 255)
    private String title;

    @Column(nullable = false, length = 50)
    private String modality;

    @Column(name = "task_type", length = 50)
    private String taskType;

    @Column(length = 50)
    private String budget;

    @Column(columnDefinition = "TEXT")
    private String description;

    @Column(nullable = false, length = 50)
    private String status;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    public ExpertCommissionEntity() {
        this.status = "OPEN";
        this.createdAt = LocalDateTime.now();
    }

    public ExpertCommissionEntity(String id, String clientName, String email, String title,
                                  String modality, String taskType, String budget,
                                  String description, String status) {
        this.id = id;
        this.clientName = clientName;
        this.email = email;
        this.title = title;
        this.modality = modality;
        this.taskType = taskType;
        this.budget = budget;
        this.description = description;
        this.status = (status != null && !status.isBlank()) ? status : "OPEN";
        this.createdAt = LocalDateTime.now();
    }

    public static ExpertCommissionEntity fromDto(ExpertCommissionDto dto) {
        ExpertCommissionEntity entity = new ExpertCommissionEntity();
        entity.setId(dto.getId());
        entity.setClientName(dto.getClientName());
        entity.setEmail(dto.getEmail());
        entity.setTitle(dto.getTitle());
        entity.setModality(dto.getModality());
        entity.setTaskType(dto.getTaskType());
        entity.setBudget(dto.getBudget());
        entity.setDescription(dto.getDescription());
        entity.setStatus(dto.getStatus() != null ? dto.getStatus() : "OPEN");
        if (dto.getCreatedAt() != null) {
            entity.setCreatedAt(dto.getCreatedAt());
        }
        return entity;
    }

    public ExpertCommissionDto toDto() {
        ExpertCommissionDto dto = new ExpertCommissionDto(
                id, clientName, email, title, modality, taskType, budget, description, status
        );
        if (createdAt != null) {
            dto.setCreatedAt(createdAt);
        }
        return dto;
    }

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getClientName() { return clientName; }
    public void setClientName(String clientName) { this.clientName = clientName; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getModality() { return modality; }
    public void setModality(String modality) { this.modality = modality; }

    public String getTaskType() { return taskType; }
    public void setTaskType(String taskType) { this.taskType = taskType; }

    public String getBudget() { return budget; }
    public void setBudget(String budget) { this.budget = budget; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
