package com.axiomcloud.dto;

import com.axiomcloud.model.TrainedModelEntity;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

import java.time.OffsetDateTime;

public class ModelResponseDto {

    private Integer id;

    @JsonProperty("job_id")
    private String jobId;

    @JsonProperty("model_name")
    private String modelName;

    @JsonProperty("model_type")
    private String modelType;

    @JsonProperty("task_type")
    private String taskType;

    @JsonProperty("is_deployed")
    private Boolean isDeployed;

    private Double accuracy;

    @JsonProperty("f1_score")
    private Double f1Score;

    @JsonProperty("roc_auc")
    private Double rocAuc;

    private Double rmse;
    private Double mae;

    @JsonProperty("r2_score")
    private Double r2Score;

    private JsonNode metrics;

    @JsonProperty("feature_importance")
    private JsonNode featureImportance;

    @JsonProperty("confusion_matrix")
    private JsonNode confusionMatrix;

    @JsonProperty("roc_curve_data")
    private JsonNode rocCurveData;

    @JsonProperty("cv_scores")
    private JsonNode cvScores;

    @JsonProperty("training_time")
    private Double trainingTime;

    @JsonProperty("created_at")
    private OffsetDateTime createdAt;

    public ModelResponseDto() {
    }

    public static ModelResponseDto fromEntity(TrainedModelEntity entity) {
        ModelResponseDto dto = new ModelResponseDto();
        dto.setId(entity.getId());
        dto.setJobId(entity.getJobId());
        dto.setModelName(entity.getModelName());
        dto.setModelType(entity.getModelType());
        dto.setTaskType(entity.getTaskType());
        dto.setIsDeployed(entity.getIsDeployed());
        dto.setAccuracy(entity.getAccuracy());
        dto.setF1Score(entity.getF1Score());
        dto.setRocAuc(entity.getRocAuc());
        dto.setRmse(entity.getRmse());
        dto.setMae(entity.getMae());
        dto.setR2Score(entity.getR2Score());
        dto.setMetrics(entity.getMetrics());
        dto.setFeatureImportance(entity.getFeatureImportance());
        dto.setConfusionMatrix(entity.getConfusionMatrix());
        dto.setRocCurveData(entity.getRocCurveData());
        dto.setCvScores(entity.getCvScores());
        dto.setTrainingTime(entity.getTrainingTime());
        dto.setCreatedAt(entity.getCreatedAt());
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

    public String getModelName() {
        return modelName;
    }

    public String getName() {
        return modelName;
    }

    public Boolean getIsChampion() {
        return isDeployed != null ? isDeployed : true;
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
