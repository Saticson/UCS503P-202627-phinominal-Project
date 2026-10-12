package com.axiomcloud.model;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.OffsetDateTime;

@Entity
@Table(name = "trained_models")
public class TrainedModelEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "job_id", length = 100)
    private String jobId;

    @Column(name = "model_name", nullable = false, length = 255)
    private String modelName;

    @Column(name = "model_type", length = 100)
    private String modelType;

    @Column(name = "task_type", length = 50)
    private String taskType;

    @Column(name = "file_path", length = 512)
    private String filePath;

    @Column(name = "is_deployed")
    private Boolean isDeployed = false;

    private Double accuracy;

    @Column(name = "f1_score")
    private Double f1Score;

    @Column(name = "roc_auc")
    private Double rocAuc;

    private Double rmse;
    private Double mae;

    @Column(name = "r2_score")
    private Double r2Score;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column
    private JsonNode metrics;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "feature_importance")
    private JsonNode featureImportance;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "confusion_matrix")
    private JsonNode confusionMatrix;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "roc_curve_data")
    private JsonNode rocCurveData;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "cv_scores")
    private JsonNode cvScores;

    @Column(name = "training_time")
    private Double trainingTime;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private OffsetDateTime createdAt;

    public TrainedModelEntity() {
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

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public String getTaskType() {
        return taskType;
    }

    public void setTaskType(String taskType) {
        this.taskType = taskType;
    }

    public String getFilePath() {
        return filePath;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    public Boolean getIsDeployed() {
        return isDeployed;
    }

    public void setIsDeployed(Boolean isDeployed) {
        this.isDeployed = isDeployed;
    }

    public Double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(Double accuracy) {
        this.accuracy = accuracy;
    }

    public Double getF1Score() {
        return f1Score;
    }

    public void setF1Score(Double f1Score) {
        this.f1Score = f1Score;
    }

    public Double getRocAuc() {
        return rocAuc;
    }

    public void setRocAuc(Double rocAuc) {
        this.rocAuc = rocAuc;
    }

    public Double getRmse() {
        return rmse;
    }

    public void setRmse(Double rmse) {
        this.rmse = rmse;
    }

    public Double getMae() {
        return mae;
    }

    public void setMae(Double mae) {
        this.mae = mae;
    }

    public Double getR2Score() {
        return r2Score;
    }

    public void setR2Score(Double r2Score) {
        this.r2Score = r2Score;
    }

    public JsonNode getMetrics() {
        return metrics;
    }

    public void setMetrics(JsonNode metrics) {
        this.metrics = metrics;
    }

    public JsonNode getFeatureImportance() {
        return featureImportance;
    }

    public void setFeatureImportance(JsonNode featureImportance) {
        this.featureImportance = featureImportance;
    }

    public JsonNode getConfusionMatrix() {
        return confusionMatrix;
    }

    public void setConfusionMatrix(JsonNode confusionMatrix) {
        this.confusionMatrix = confusionMatrix;
    }

    public JsonNode getRocCurveData() {
        return rocCurveData;
    }

    public void setRocCurveData(JsonNode rocCurveData) {
        this.rocCurveData = rocCurveData;
    }

    public JsonNode getCvScores() {
        return cvScores;
    }

    public void setCvScores(JsonNode cvScores) {
        this.cvScores = cvScores;
    }

    public Double getTrainingTime() {
        return trainingTime;
    }

    public void setTrainingTime(Double trainingTime) {
        this.trainingTime = trainingTime;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(OffsetDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
