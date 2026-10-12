package com.axiomcloud.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public class PredictionRequestDto {

    @JsonProperty("model_id")
    private Integer modelId;

    private Map<String, Object> features;

    public PredictionRequestDto() {
    }

    public Integer getModelId() {
        return modelId;
    }

    public void setModelId(Integer modelId) {
        this.modelId = modelId;
    }

    public Map<String, Object> getFeatures() {
        return features;
    }

    public void setFeatures(Map<String, Object> features) {
        this.features = features;
    }
}
