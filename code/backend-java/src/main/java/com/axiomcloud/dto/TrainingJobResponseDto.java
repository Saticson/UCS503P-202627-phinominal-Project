package com.axiomcloud.dto;

import com.axiomcloud.model.TrainingJobEntity;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

import java.time.OffsetDateTime;
import java.util.Map;

public class TrainingJobResponseDto {

    private Integer id;

    @JsonProperty("job_id")
    private String jobId;

    @JsonProperty("dataset_id")
    private Integer datasetId;

    @JsonProperty("task_type")
    private String taskType;

    @JsonProperty("target_column")
    private String targetColumn;

    private String status;
    private Integer progress;

    @JsonProperty("error_message")
    private String errorMessage;

    @JsonProperty("created_at")
    private OffsetDateTime createdAt;

    @JsonProperty("completed_at")
    private OffsetDateTime completedAt;

    @JsonProperty("execution_mode")
    private String executionMode;

    @JsonProperty("local_job_spec")
    private Map<String, Object> localJobSpec;

    private String message;

    @JsonProperty("best_model")
    private String bestModel;

    @JsonProperty("best_score")
    private Double bestScore;

    public TrainingJobResponseDto() {
    }

    public static TrainingJobResponseDto fromEntity(TrainingJobEntity entity) {
        TrainingJobResponseDto dto = new TrainingJobResponseDto();
        dto.setId(entity.getId());
        dto.setJobId(entity.getJobId());
        dto.setDatasetId(entity.getDatasetId());
        dto.setTaskType(entity.getTaskType());
        dto.setTargetColumn(entity.getTargetColumn());
        dto.setStatus(entity.getStatus());
        dto.setProgress(entity.getProgress());
        dto.setErrorMessage(entity.getErrorMessage());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setCompletedAt(entity.getCompletedAt());
        return dto;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getJobId() {
        return jobId;
    }

    public void setJobId(String jobId) {
        this.jobId = jobId;
    }

    public Integer getDatasetId() {
        return datasetId;
    }

    public void setDatasetId(Integer datasetId) {
        this.datasetId = datasetId;
    }

    public String getTaskType() {
        return taskType;
    }

    public void setTaskType(String taskType) {
        this.taskType = taskType;
    }

    public String getTargetColumn() {
        return targetColumn;
    }

    public void setTargetColumn(String targetColumn) {
        this.targetColumn = targetColumn;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Integer getProgress() {
        return progress;
    }

    public void setProgress(Integer progress) {
        this.progress = progress;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(OffsetDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public OffsetDateTime getCompletedAt() {
        return completedAt;
    }

    public void setCompletedAt(OffsetDateTime completedAt) {
        this.completedAt = completedAt;
    }

    public String getExecutionMode() {
        return executionMode;
    }

    public void setExecutionMode(String executionMode) {
        this.executionMode = executionMode;
    }

    public Map<String, Object> getLocalJobSpec() {
        return localJobSpec;
    }

    public void setLocalJobSpec(Map<String, Object> localJobSpec) {
        this.localJobSpec = localJobSpec;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getBestModel() {
        return bestModel;
    }

    public void setBestModel(String bestModel) {
        this.bestModel = bestModel;
    }

    public Double getBestScore() {
        return bestScore;
    }

    public void setBestScore(Double bestScore) {
        this.bestScore = bestScore;
    }
}
