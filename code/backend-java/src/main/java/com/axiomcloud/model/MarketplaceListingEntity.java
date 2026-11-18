package com.axiomcloud.model;

import com.axiomcloud.dto.MarketplaceListingDto;
import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Entity
@Table(name = "marketplace_listings")
public class MarketplaceListingEntity {

    @Id
    @Column(nullable = false, length = 100)
    private String id;

    @Column(nullable = false, length = 255)
    private String title;

    @Column(columnDefinition = "TEXT")
    private String description;

    @Column(nullable = false, length = 50)
    private String modality;

    @Column(name = "task_type", length = 50)
    private String taskType;

    @Column(length = 50)
    private String framework;

    @Column(name = "metric_score")
    private double metricScore;

    @Column(name = "metric_name", length = 50)
    private String metricName;

    @Column(name = "latency_ms")
    private int latencyMs;

    @Column(length = 50)
    private String price;

    @Column(name = "is_verified")
    private boolean verified;

    @Column(name = "seller_name", length = 255)
    private String sellerName;

    @Column(name = "download_url", length = 512)
    private String downloadUrl;

    @Column(name = "input_shape", length = 100)
    private String inputShape;

    @Column(name = "license_type", length = 100)
    private String license;

    @Column(name = "tags_csv", length = 512)
    private String tagsCsv;

    @Column(name = "artifact_file_path", length = 512)
    private String artifactFilePath;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private OffsetDateTime createdAt;

    public MarketplaceListingEntity() {}

    public static MarketplaceListingEntity fromDto(MarketplaceListingDto dto) {
        MarketplaceListingEntity entity = new MarketplaceListingEntity();
        entity.setId(dto.getId());
        entity.setTitle(dto.getTitle());
        entity.setDescription(dto.getDescription());
        entity.setModality(dto.getModality());
        entity.setTaskType(dto.getTaskType());
        entity.setFramework(dto.getFramework());
        entity.setMetricScore(dto.getMetricScore());
        entity.setMetricName(dto.getMetricName());
        entity.setLatencyMs(dto.getLatencyMs());
        entity.setPrice(dto.getPrice());
        entity.setVerified(dto.isVerified());
        entity.setSellerName(dto.getSellerName());
        entity.setDownloadUrl(dto.getDownloadUrl());
        entity.setInputShape(dto.getInputShape());
        entity.setLicense(dto.getLicense());
        if (dto.getTags() != null) {
            entity.setTagsCsv(String.join(",", dto.getTags()));
        }
        return entity;
    }

    public MarketplaceListingDto toDto() {
        List<String> tagsList = new ArrayList<>();
        if (tagsCsv != null && !tagsCsv.isBlank()) {
            tagsList = Arrays.asList(tagsCsv.split("\\s*,\\s*"));
        }
        MarketplaceListingDto dto = new MarketplaceListingDto(
                id, title, description, modality, taskType, framework,
                metricScore, metricName, latencyMs, price, verified,
                sellerName, inputShape, license, tagsList
        );
        dto.setDownloadUrl(downloadUrl != null ? downloadUrl : "/api/marketplace/" + id + "/download");
        return dto;
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

    public String getTagsCsv() { return tagsCsv; }
    public void setTagsCsv(String tagsCsv) { this.tagsCsv = tagsCsv; }

    public String getArtifactFilePath() { return artifactFilePath; }
    public void setArtifactFilePath(String artifactFilePath) { this.artifactFilePath = artifactFilePath; }

    public OffsetDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(OffsetDateTime createdAt) { this.createdAt = createdAt; }
}
