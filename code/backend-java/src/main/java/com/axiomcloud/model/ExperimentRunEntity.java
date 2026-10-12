package com.axiomcloud.model;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.OffsetDateTime;

@Entity
@Table(name = "experiment_runs")
public class ExperimentRunEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "run_id", nullable = false, unique = true, length = 100)
    private String runId;

    @Column(name = "job_id", length = 100)
    private String jobId;

    @Column(name = "dataset_id")
    private Integer datasetId;

    @Column(name = "target_column", length = 255)
    private String targetColumn;

    @Column(name = "task_type", length = 50)
    private String taskType;

    @Column(length = 50)
    private String status = "pending";

    @JdbcTypeCode(SqlTypes.JSON)
    @Column
    private JsonNode config;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "summary_metrics")
    private JsonNode summaryMetrics;

    @Column(name = "best_model_name", length = 255)
    private String bestModelName;

    @Column(name = "best_score")
    private Double bestScore;

    @Column(name = "error_message", columnDefinition = "text")
    private String errorMessage;

    @CreationTimestamp
    @Column(name = "started_at", updatable = false)
    private OffsetDateTime startedAt;

    @Column(name = "completed_at")
    private OffsetDateTime completedAt;

    public ExperimentRunEntity() {
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
