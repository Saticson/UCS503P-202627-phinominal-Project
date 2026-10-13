package com.axiomcloud.dto;

import java.util.List;

public class MarketplaceListingDto {
    private String id;
    private String title;
    private String description;
    private String modality; // "COMPUTER_VISION" or "TABULAR"
    private String taskType; // "CLASSIFICATION", "DETECTION", "REGRESSION"
    private String framework; // "PyTorch", "ONNX", "Scikit-Learn", "TensorFlow"
    private double metricScore;
    private String metricName; // "Accuracy", "F1", "mAP", "AUROC"
    private int latencyMs;
    private String price; // "Free", "$29", "$49"
    private boolean verified;
    private String sellerName;
    private String downloadUrl;
    private String inputShape;
    private String license;
    private List<String> tags;

    public MarketplaceListingDto() {}

    public MarketplaceListingDto(String id, String title, String description, String modality, 
                                String taskType, String framework, double metricScore, 
                                String metricName, int latencyMs, String price, boolean verified, 
                                String sellerName, String inputShape, String license, List<String> tags) {
        this.id = id;
        this.title = title;
        this.description = description;
        this.modality = modality;
        this.taskType = taskType;
        this.framework = framework;
        this.metricScore = metricScore;
        this.metricName = metricName;
        this.latencyMs = latencyMs;
        this.price = price;
        this.verified = verified;
        this.sellerName = sellerName;
        this.downloadUrl = "/api/marketplace/" + id + "/download";
        this.inputShape = inputShape;
        this.license = license;
        this.tags = tags;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getModality() { return modality; }
    public void setModality(String modality) { this.modality = modality; }

    public String getTaskType() { return taskType; }
    public void setTaskType(String taskType) { this.taskType = taskType; }

    public String getFramework() { return framework; }
    public void setFramework(String framework) { this.framework = framework; }

    public double getMetricScore() { return metricScore; }
    public void setMetricScore(double metricScore) { this.metricScore = metricScore; }

    public String getMetricName() { return metricName; }
    public void setMetricName(String metricName) { this.metricName = metricName; }

    public int getLatencyMs() { return latencyMs; }
    public void setLatencyMs(int latencyMs) { this.latencyMs = latencyMs; }

    public String getPrice() { return price; }
    public void setPrice(String price) { this.price = price; }

    public boolean isVerified() { return verified; }
    public void setVerified(boolean verified) { this.verified = verified; }

    public String getSellerName() { return sellerName; }
    public void setSellerName(String sellerName) { this.sellerName = sellerName; }

    public String getDownloadUrl() { return downloadUrl; }
    public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }

    public String getInputShape() { return inputShape; }
    public void setInputShape(String inputShape) { this.inputShape = inputShape; }

    public String getLicense() { return license; }
    public void setLicense(String license) { this.license = license; }

    public List<String> getTags() { return tags; }
    public void setTags(List<String> tags) { this.tags = tags; }
}
