package com.axiomcloud.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class DashboardSummaryDto {

    @JsonProperty("total_datasets")
    private long totalDatasets;

    @JsonProperty("total_models")
    private long totalModels;

    @JsonProperty("deployed_models")
    private long deployedModels;

    @JsonProperty("completed_jobs")
    private long completedJobs;

    @JsonProperty("failed_jobs")
    private long failedJobs;

    public DashboardSummaryDto() {
    }

    public DashboardSummaryDto(long totalDatasets, long totalModels, long deployedModels, long completedJobs, long failedJobs) {
        this.totalDatasets = totalDatasets;
        this.totalModels = totalModels;
        this.deployedModels = deployedModels;
        this.completedJobs = completedJobs;
        this.failedJobs = failedJobs;
    }

    public long getTotalDatasets() {
        return totalDatasets;
    }

    public void setTotalDatasets(long totalDatasets) {
        this.totalDatasets = totalDatasets;
    }

    public long getTotalModels() {
        return totalModels;
    }

    public void setTotalModels(long totalModels) {
        this.totalModels = totalModels;
    }

    public long getDeployedModels() {
        return deployedModels;
    }

    public void setDeployedModels(long deployedModels) {
        this.deployedModels = deployedModels;
    }

    public long getCompletedJobs() {
        return completedJobs;
    }

    public void setCompletedJobs(long completedJobs) {
        this.completedJobs = completedJobs;
    }

    public long getFailedJobs() {
        return failedJobs;
    }

    public void setFailedJobs(long failedJobs) {
        this.failedJobs = failedJobs;
    }
}
