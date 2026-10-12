package com.axiomcloud.service;

import com.axiomcloud.dto.ModelResponseDto;
import com.axiomcloud.model.DatasetEntity;
import com.axiomcloud.model.TrainedModelEntity;
import com.axiomcloud.model.TrainingJobEntity;
import com.axiomcloud.repository.DatasetRepository;
import com.axiomcloud.repository.TrainedModelRepository;
import com.axiomcloud.repository.TrainingJobRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.io.File;
import java.util.*;

@Service
public class ModelService {

    private static final Logger logger = LoggerFactory.getLogger(ModelService.class);

    private final TrainedModelRepository trainedModelRepository;
    private final TrainingJobRepository trainingJobRepository;
    private final DatasetRepository datasetRepository;
    private final MlEngineClient mlEngineClient;
    private final ObjectMapper objectMapper;

    public ModelService(TrainedModelRepository trainedModelRepository,
                        TrainingJobRepository trainingJobRepository,
                        DatasetRepository datasetRepository,
                        MlEngineClient mlEngineClient,
                        ObjectMapper objectMapper) {
        this.trainedModelRepository = trainedModelRepository;
        this.trainingJobRepository = trainingJobRepository;
        this.datasetRepository = datasetRepository;
        this.mlEngineClient = mlEngineClient;
        this.objectMapper = objectMapper;
    }

    public List<ModelResponseDto> listModels(String jobId) {
        List<TrainedModelEntity> models = (jobId != null && !jobId.isBlank())
                ? trainedModelRepository.findByJobIdOrderByCreatedAtDesc(jobId)
                : trainedModelRepository.findAllByOrderByCreatedAtDesc();

        return models.stream()
                .map(ModelResponseDto::fromEntity)
                .toList();
    }

    public ModelResponseDto getModel(int modelId) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));
        return ModelResponseDto.fromEntity(model);
    }

    @Transactional
    public Map<String, Object> deployModel(int modelId) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));

        if (model.getJobId() != null) {
            trainedModelRepository.undeployOtherModelsInJob(model.getJobId(), modelId);
        }

        model.setIsDeployed(true);
        trainedModelRepository.save(model);
        logger.info("Model {} ({}) deployed successfully", modelId, model.getModelName());

        return Map.of(
                "message", "Model '" + model.getModelName() + "' deployed successfully",
                "model_id", modelId
        );
    }

    @Transactional
    public Map<String, Object> undeployModel(int modelId) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));

        model.setIsDeployed(false);
        trainedModelRepository.save(model);
        return Map.of("message", "Model undeployed", "model_id", modelId);
    }

    @Transactional
    public void deleteModel(int modelId) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));

        if (model.getFilePath() != null) {
            new File(model.getFilePath()).delete();
        }
        trainedModelRepository.delete(model);
    }

    public Resource downloadModelFile(int modelId) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));

        if (model.getFilePath() == null || !new File(model.getFilePath()).exists()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Model file not found on disk");
        }
        return new FileSystemResource(model.getFilePath());
    }

    public List<Map<String, Object>> listDeployedModels() {
        return trainedModelRepository.findByIsDeployedTrue()
                .stream()
                .map(m -> Map.<String, Object>of(
                        "id", m.getId(),
                        "model_name", m.getModelName(),
                        "model_type", m.getModelType() != null ? m.getModelType() : "",
                        "task_type", m.getTaskType() != null ? m.getTaskType() : "",
                        "job_id", m.getJobId() != null ? m.getJobId() : ""
                ))
                .toList();
    }

    public Map<String, Object> getInferenceTemplate(int modelId, boolean randomize) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));

        TrainingJobEntity job = trainingJobRepository.findByJobId(model.getJobId())
                .orElse(null);

        DatasetEntity dataset = job != null && job.getDatasetId() != null
                ? datasetRepository.findById(job.getDatasetId()).orElse(null)
                : null;

        List<Map<String, Object>> features = new ArrayList<>();

        if (dataset != null && dataset.getColumnsInfo() != null && dataset.getColumnsInfo().isArray()) {
            for (JsonNode col : dataset.getColumnsInfo()) {
                String name = col.has("name") ? col.get("name").asText() : "";
                if (name.equals(job != null ? job.getTargetColumn() : "")) continue;

                String dtype = col.has("dtype") ? col.get("dtype").asText() : "object";
                boolean isNumeric = dtype.contains("int") || dtype.contains("float");

                Map<String, Object> feat = new HashMap<>();
                feat.put("name", name);
                feat.put("dtype", dtype);
                feat.put("input_type", isNumeric ? "number" : "select");

                if (isNumeric) {
                    double min = col.has("stats") && col.get("stats").has("min") ? col.get("stats").get("min").asDouble() : 0.0;
                    double max = col.has("stats") && col.get("stats").has("max") ? col.get("stats").get("max").asDouble() : 100.0;
                    double mean = col.has("stats") && col.get("stats").has("mean") ? col.get("stats").get("mean").asDouble() : 50.0;
                    feat.put("min", min);
                    feat.put("max", max);
                    feat.put("default_value", randomize ? (min + Math.random() * (max - min)) : mean);
                } else {
                    List<String> options = new ArrayList<>();
                    if (col.has("sample_values") && col.get("sample_values").isArray()) {
                        for (JsonNode v : col.get("sample_values")) options.add(v.asText());
                    }
                    feat.put("options", options);
                    feat.put("default_value", options.isEmpty() ? "" : options.get(0));
                }
                features.add(feat);
            }
        }

        Map<String, Object> result = new HashMap<>();
        result.put("model_id", model.getId());
        result.put("model_name", model.getModelName());
        result.put("task_type", model.getTaskType());
        result.put("job_id", model.getJobId());
        result.put("dataset_id", dataset != null ? dataset.getId() : null);
        result.put("dataset_name", dataset != null ? dataset.getName() : null);
        result.put("target_column", job != null ? job.getTargetColumn() : null);
        result.put("features", features);
        return result;
    }

    public JsonNode getShapExplanation(int modelId, int sampleIndex, int nsamples) {
        Map<String, Object> payload = explainabilityPayload(modelId);
        payload.put("sample_index", sampleIndex);
        payload.put("nsamples", nsamples);
        return mlEngineClient.explainShap(payload);
    }

    public JsonNode getLimeExplanation(int modelId, int sampleIndex, int numFeatures, Map<String, Object> customInput) {
        Map<String, Object> payload = explainabilityPayload(modelId);
        payload.put("sample_index", sampleIndex);
        payload.put("num_features", numFeatures);
        payload.put("custom_input", customInput);
        return mlEngineClient.explainLime(payload);
    }

    public Map<String, Object> getMonitoring(int modelId, Integer compareDatasetId) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));
        TrainingJobEntity job = trainingJobRepository.findByJobId(model.getJobId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Training job not found"));

        List<Double> scores = new ArrayList<>();
        if (model.getCvScores() != null && model.getCvScores().isArray()) {
            for (JsonNode value : model.getCvScores()) {
                if (value.isNumber()) scores.add(value.asDouble());
            }
        }
        Double mean = scores.isEmpty() ? null : scores.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        Double variance = scores.isEmpty() ? null : scores.stream()
                .mapToDouble(score -> Math.pow(score - mean, 2)).average().orElse(0);
        Double std = variance == null ? null : Math.sqrt(variance);

        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("model_id", model.getId());
        snapshot.put("model_name", model.getModelName());
        snapshot.put("task_type", model.getTaskType());
        snapshot.put("is_deployed", model.getIsDeployed());
        snapshot.put("training_time", model.getTrainingTime());
        snapshot.put("primary_metric", "classification".equalsIgnoreCase(model.getTaskType()) ? model.getAccuracy() : model.getR2Score());
        snapshot.put("cv_mean", mean);
        snapshot.put("cv_std", std);
        snapshot.put("stability", std == null ? "unknown" : std < 0.03 ? "stable" : std < 0.08 ? "moderate" : "unstable");

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("monitoring", snapshot);
        result.put("drift", null);
        if (compareDatasetId != null) {
            DatasetEntity trainingDataset = datasetRepository.findById(job.getDatasetId())
                    .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Training dataset not found"));
            DatasetEntity comparisonDataset = datasetRepository.findById(compareDatasetId)
                    .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Comparison dataset not found"));
            result.put("drift", objectMapper.convertValue(
                    mlEngineClient.monitoring(trainingDataset.getFilePath(), comparisonDataset.getFilePath(), job.getTargetColumn()),
                    Map.class));
        }
        return result;
    }

    private Map<String, Object> explainabilityPayload(int modelId) {
        TrainedModelEntity model = trainedModelRepository.findById(modelId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));
        TrainingJobEntity job = trainingJobRepository.findByJobId(model.getJobId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Training job not found"));
        DatasetEntity dataset = datasetRepository.findById(job.getDatasetId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Training dataset not found"));

        if (model.getFilePath() == null || !new File(model.getFilePath()).isFile()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Model artifact not found");
        }

        Map<String, Object> payload = new HashMap<>();
        payload.put("model_id", model.getId());
        payload.put("model_name", model.getModelName());
        payload.put("model_type", model.getModelType());
        payload.put("task_type", model.getTaskType());
        payload.put("job_id", model.getJobId());
        payload.put("file_path", model.getFilePath());
        payload.put("dataset_path", dataset.getFilePath());
        payload.put("target_column", job.getTargetColumn());
        return payload;
    }
}
