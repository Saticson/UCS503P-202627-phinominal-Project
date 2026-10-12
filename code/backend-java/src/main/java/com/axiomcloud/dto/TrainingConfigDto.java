package com.axiomcloud.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

import java.util.List;
import java.util.Map;

public class TrainingConfigDto {

    @JsonProperty("dataset_id")
    @NotNull
    @Min(1)
    private Integer datasetId;

    @JsonProperty("target_column")
    @NotBlank
    private String targetColumn;

    @JsonProperty("execution_mode")
    @Pattern(regexp = "remote|local", message = "execution_mode must be remote or local")
    private String executionMode = "remote";

    @JsonProperty("task_type")
    @Pattern(regexp = "auto|classification|regression", message = "task_type must be auto, classification, or regression")
    private String taskType;

    @JsonProperty("models_to_train")
    @Size(max = 5, message = "At most 5 models may be trained per run")
    private List<String> modelsToTrain;

    @JsonProperty("model_hyperparams")
    private Map<String, Object> modelHyperparams;

    @JsonProperty("test_size")
    @DecimalMin(value = "0.1")
    @DecimalMax(value = "0.4")
    private Double testSize = 0.2;

    @JsonProperty("cv_folds")
    @Min(2)
    @Max(5)
    private Integer cvFolds = 5;

    @JsonProperty("enable_tuning")
    private Boolean enableTuning = false;

    @JsonProperty("tuning_trials")
    @Min(1)
    @Max(100)
    private Integer tuningTrials = 12;

    @JsonProperty("tuning_time_budget_sec")
    @Min(10)
    @Max(3600)
    private Integer tuningTimeBudgetSec = 180;

    public TrainingConfigDto() {
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

    public String getExecutionMode() {
        return executionMode;
    }

    public void setExecutionMode(String executionMode) {
        this.executionMode = executionMode;
    }

    public String getTaskType() {
        return taskType;
    }

    public void setTaskType(String taskType) {
        this.taskType = taskType;
    }

    public List<String> getModelsToTrain() {
        return modelsToTrain;
    }

    public void setModelsToTrain(List<String> modelsToTrain) {
        this.modelsToTrain = modelsToTrain;
    }

    public Map<String, Object> getModelHyperparams() {
        return modelHyperparams;
    }

    public void setModelHyperparams(Map<String, Object> modelHyperparams) {
        this.modelHyperparams = modelHyperparams;
    }

    public Double getTestSize() {
        return testSize;
    }

    public void setTestSize(Double testSize) {
        this.testSize = testSize;
    }

    public Integer getCvFolds() {
        return cvFolds;
    }

    public void setCvFolds(Integer cvFolds) {
        this.cvFolds = cvFolds;
    }

    public Boolean getEnableTuning() {
        return enableTuning;
    }

    public void setEnableTuning(Boolean enableTuning) {
        this.enableTuning = enableTuning;
    }

    public Integer getTuningTrials() {
        return tuningTrials;
    }

    public void setTuningTrials(Integer tuningTrials) {
        this.tuningTrials = tuningTrials;
    }

    public Integer getTuningTimeBudgetSec() {
        return tuningTimeBudgetSec;
    }

    public void setTuningTimeBudgetSec(Integer tuningTimeBudgetSec) {
        this.tuningTimeBudgetSec = tuningTimeBudgetSec;
    }
}
