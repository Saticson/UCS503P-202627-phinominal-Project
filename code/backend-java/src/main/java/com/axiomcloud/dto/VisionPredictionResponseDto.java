package com.axiomcloud.dto;

import java.util.Map;

public class VisionPredictionResponseDto {
    private String predictedClass;
    private double confidence;
    private int latencyMs;
    private Map<String, Double> topProbabilities;
    private String imagePreview;
    private String modelName;
    private String status;

    public VisionPredictionResponseDto() {
        this.status = "SUCCESS";
    }

    public VisionPredictionResponseDto(String predictedClass, double confidence, int latencyMs,
                                       Map<String, Double> topProbabilities, String imagePreview,
                                       String modelName) {
        this.predictedClass = predictedClass;
        this.confidence = confidence;
        this.latencyMs = latencyMs;
        this.topProbabilities = topProbabilities;
        this.imagePreview = imagePreview;
        this.modelName = modelName;
        this.status = "SUCCESS";
    }

    public String getPredictedClass() { return predictedClass; }
    public void setPredictedClass(String predictedClass) { this.predictedClass = predictedClass; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }

    public int getLatencyMs() { return latencyMs; }
    public void setLatencyMs(int latencyMs) { this.latencyMs = latencyMs; }

    public Map<String, Double> getTopProbabilities() { return topProbabilities; }
    public void setTopProbabilities(Map<String, Double> topProbabilities) { this.topProbabilities = topProbabilities; }

    public String getImagePreview() { return imagePreview; }
    public void setImagePreview(String imagePreview) { this.imagePreview = imagePreview; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}
