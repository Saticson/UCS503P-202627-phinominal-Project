package com.axiomcloud.dto;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class NlpPredictionResponseDto {
    private String prediction;
    private double confidence;
    private int latencyMs;
    private Map<String, Double> probabilities;
    private List<Map<String, Object>> tokenAttributions;
    private Set<String> conformalPredictionSet;
    private String modelName;
    private String status;

    public NlpPredictionResponseDto() {
        this.status = "SUCCESS";
    }

    public NlpPredictionResponseDto(String prediction, double confidence, int latencyMs,
                                   Map<String, Double> probabilities,
                                   List<Map<String, Object>> tokenAttributions,
                                   Set<String> conformalPredictionSet,
                                   String modelName) {
        this.prediction = prediction;
        this.confidence = confidence;
        this.latencyMs = latencyMs;
        this.probabilities = probabilities;
        this.tokenAttributions = tokenAttributions;
        this.conformalPredictionSet = conformalPredictionSet;
        this.modelName = modelName;
        this.status = "SUCCESS";
    }

    public String getPrediction() { return prediction; }
    public void setPrediction(String prediction) { this.prediction = prediction; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }

    public int getLatencyMs() { return latencyMs; }
    public void setLatencyMs(int latencyMs) { this.latencyMs = latencyMs; }

    public Map<String, Double> getProbabilities() { return probabilities; }
    public void setProbabilities(Map<String, Double> probabilities) { this.probabilities = probabilities; }

    public List<Map<String, Object>> getTokenAttributions() { return tokenAttributions; }
    public void setTokenAttributions(List<Map<String, Object>> tokenAttributions) { this.tokenAttributions = tokenAttributions; }

    public Set<String> getConformalPredictionSet() { return conformalPredictionSet; }
    public void setConformalPredictionSet(Set<String> conformalPredictionSet) { this.conformalPredictionSet = conformalPredictionSet; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}
