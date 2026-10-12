package com.axiomcloud.dto;

import com.axiomcloud.model.ExperimentRunEntity;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

import java.time.OffsetDateTime;

public class ExperimentRunResponseDto {

    private Integer id;

    @JsonProperty("run_id")
    private String runId;

    @JsonProperty("job_id")
    private String jobId;

    @JsonProperty("dataset_id")
    private Integer datasetId;

    @JsonProperty("target_column")
    private String targetColumn;

    @JsonProperty("task_type")
    private String taskType;

    private String status;
    private JsonNode config;

    @JsonProperty("summary_metrics")
    private JsonNode summaryMetrics;

    @JsonProperty("best_model_name")
    private String bestModelName;

    @JsonProperty("best_score")
    private Double bestScore;

    @JsonProperty("error_message")
    private String errorMessage;

    @JsonProperty("started_at")
    private OffsetDateTime startedAt;

    @JsonProperty("completed_at")
    private OffsetDateTime completedAt;

    public ExperimentRunResponseDto() {
    }

    public static ExperimentRunResponseDto fromEntity(ExperimentRunEntity entity) {
        ExperimentRunResponseDto dto = new ExperimentRunResponseDto();
        dto.setId(entity.getId());
        dto.setRunId(entity.getRunId());
        dto.setJobId(entity.getJobId());
        dto.setDatasetId(entity.getDatasetId());
        dto.setTargetColumn(entity.getTargetColumn());
        dto.setTaskType(entity.getTaskType());
        dto.setStatus(entity.getStatus());
        dto.setConfig(entity.getConfig());
        dto.setSummaryMetrics(entity.getSummaryMetrics());
        dto.setBestModelName(entity.getBestModelName());
        dto.setBestScore(entity.getBestScore());
        dto.setErrorMessage(entity.getErrorMessage());
        dto.setStartedAt(entity.getStartedAt());
        dto.setCompletedAt(entity.getCompletedAt());
        return dto;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getRunId() {
        return runId;
    }

    public void setRunId(String runId) {
        this.runId = runId;
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

    public String getTargetColumn() {
        return targetColumn;
    }

    public void setTargetColumn(String targetColumn) {
        this.targetColumn = targetColumn;
    }

    public String getTaskType() {
        return taskType;
    }

    public void setTaskType(String taskType) {
        this.taskType = taskType;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public JsonNode getConfig() {
        return config;
    }

    public void setConfig(JsonNode config) {
        this.config = config;
    }

    public JsonNode getSummaryMetrics() {
        return summaryMetrics;
    }

    public void setSummaryMetrics(JsonNode summaryMetrics) {
        this.summaryMetrics = summaryMetrics;
    }

    public String getBestModelName() {
        return bestModelName;
    }

    public void setBestModelName(String bestModelName) {
        this.bestModelName = bestModelName;
    }

    public Double getBestScore() {
        return bestScore;
    }

    public void setBestScore(Double bestScore) {
        this.bestScore = bestScore;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public OffsetDateTime getStartedAt() {
        return startedAt;
    }

    public void setStartedAt(OffsetDateTime startedAt) {
        this.startedAt = startedAt;
    }

    public OffsetDateTime getCompletedAt() {
        return completedAt;
    }

    public void setCompletedAt(OffsetDateTime completedAt) {
        this.completedAt = completedAt;
    }
}
