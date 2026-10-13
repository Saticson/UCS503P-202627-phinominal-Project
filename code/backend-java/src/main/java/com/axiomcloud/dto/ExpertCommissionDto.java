package com.axiomcloud.dto;

import java.time.LocalDateTime;

public class ExpertCommissionDto {
    private String id;
    private String clientName;
    private String email;
    private String title;
    private String modality; // "COMPUTER_VISION", "TABULAR"
    private String taskType;
    private String budget;
    private String description;
    private String status; // "OPEN", "MATCHED", "IN_PROGRESS"
    private LocalDateTime createdAt;

    public ExpertCommissionDto() {
        this.createdAt = LocalDateTime.now();
        this.status = "OPEN";
    }

    public ExpertCommissionDto(String id, String clientName, String email, String title, 
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
        this.status = status;
        this.createdAt = LocalDateTime.now();
    }

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
