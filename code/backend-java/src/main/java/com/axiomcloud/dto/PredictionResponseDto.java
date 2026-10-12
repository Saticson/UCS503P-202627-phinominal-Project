package com.axiomcloud.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public class PredictionResponseDto {

    @JsonProperty("model_id")
    private Integer modelId;

    @JsonProperty("model_name")
    private String modelName;

    private Object prediction;

    private Map<String, Double> probability;

    private Double confidence;

    public PredictionResponseDto() {
    }

    public Integer getModelId() {
        return modelId;
    }

    public void setModelId(Integer modelId) {
        this.modelId = modelId;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public Object getPrediction() {
        return prediction;
    }

    public void setPrediction(Object prediction) {
        this.prediction = prediction;
    }

    public Map<String, Double> getProbability() {
        return probability;
    }

    public void setProbability(Map<String, Double> probability) {
        this.probability = probability;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }
}
