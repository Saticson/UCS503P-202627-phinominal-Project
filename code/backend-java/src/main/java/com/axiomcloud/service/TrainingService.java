package com.axiomcloud.service;

import com.axiomcloud.dto.TrainingConfigDto;
import com.axiomcloud.dto.TrainingJobResponseDto;
import com.axiomcloud.model.DatasetEntity;
import com.axiomcloud.model.ExperimentRunEntity;
import com.axiomcloud.model.TrainedModelEntity;
import com.axiomcloud.model.TrainingJobEntity;
import com.axiomcloud.repository.DatasetRepository;
import com.axiomcloud.repository.ExperimentRunRepository;
import com.axiomcloud.repository.TrainedModelRepository;
import com.axiomcloud.repository.TrainingJobRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.task.TaskExecutor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.web.server.ResponseStatusException;

import java.time.OffsetDateTime;
import java.util.*;

@Service
public class TrainingService {

    private static final Logger logger = LoggerFactory.getLogger(TrainingService.class);

    private final TrainingJobRepository trainingJobRepository;
    private final ExperimentRunRepository experimentRunRepository;
    private final TrainedModelRepository trainedModelRepository;
    private final DatasetRepository datasetRepository;
    private final MlEngineClient mlEngineClient;
    private final ObjectMapper objectMapper;
    private final TaskExecutor trainingExecutor;

    public TrainingService(TrainingJobRepository trainingJobRepository,
                           ExperimentRunRepository experimentRunRepository,
                           TrainedModelRepository trainedModelRepository,
                           DatasetRepository datasetRepository,
                           MlEngineClient mlEngineClient,
                           ObjectMapper objectMapper,
                           @Qualifier("trainingExecutor") TaskExecutor trainingExecutor) {
        this.trainingJobRepository = trainingJobRepository;
        this.experimentRunRepository = experimentRunRepository;
        this.trainedModelRepository = trainedModelRepository;
        this.datasetRepository = datasetRepository;
        this.mlEngineClient = mlEngineClient;
        this.objectMapper = objectMapper;
        this.trainingExecutor = trainingExecutor;
    }

    @Transactional
    public TrainingJobResponseDto launchTraining(TrainingConfigDto config) {
        DatasetEntity dataset = datasetRepository.findById(config.getDatasetId())
                .orElseGet(() -> datasetRepository.findAll().stream().findFirst()
                        .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Dataset not found")));

        String jobId = UUID.randomUUID().toString();
        String runId = UUID.randomUUID().toString();
        boolean isLocal = "local".equalsIgnoreCase(config.getExecutionMode());

        TrainingJobEntity job = new TrainingJobEntity();
        job.setJobId(jobId);
        job.setDatasetId(dataset.getId());
        job.setTaskType(config.getTaskType());
        job.setTargetColumn(config.getTargetColumn());
        job.setStatus(isLocal ? "local_pending" : "pending");
        job.setProgress(0);

        Map<String, Object> configMap = new HashMap<>();
        configMap.put("dataset_path", dataset.getFilePath());
        configMap.put("target_column", config.getTargetColumn());
        configMap.put("task_type", config.getTaskType());
        configMap.put("execution_mode", config.getExecutionMode());
        configMap.put("test_size", config.getTestSize());
        configMap.put("cv_folds", config.getCvFolds());
        configMap.put("models_to_train", config.getModelsToTrain());
        configMap.put("enable_tuning", config.getEnableTuning());
        configMap.put("tuning_trials", config.getTuningTrials());
        configMap.put("tuning_time_budget_sec", config.getTuningTimeBudgetSec());

        Map<String, Object> localJobSpec = null;
        if (isLocal) {
            localJobSpec = new HashMap<>();
            localJobSpec.put("job_id", jobId);
            localJobSpec.put("run_id", runId);
            localJobSpec.put("dataset_id", dataset.getId());
            localJobSpec.put("dataset_name", dataset.getName());
            localJobSpec.put("dataset_path", dataset.getFilePath());
            localJobSpec.put("target_column", config.getTargetColumn());
            localJobSpec.put("task_type", config.getTaskType());
            localJobSpec.put("output_dir", "./storage/models/local_runs/" + jobId);
            localJobSpec.put("hyperparameters", Map.of(
                    "test_size", config.getTestSize(),
                    "cv_folds", config.getCvFolds()
            ));
            configMap.put("local_job_spec", localJobSpec);
        }

        job.setConfig(objectMapper.valueToTree(configMap));
        TrainingJobEntity savedJob = trainingJobRepository.save(job);

        ExperimentRunEntity run = new ExperimentRunEntity();
        run.setRunId(runId);
        run.setJobId(jobId);
        run.setDatasetId(dataset.getId());
        run.setTargetColumn(config.getTargetColumn());
        run.setTaskType(config.getTaskType());
        run.setStatus(isLocal ? "local_pending" : "pending");
        run.setConfig(objectMapper.valueToTree(configMap));
        experimentRunRepository.save(run);

        TrainingJobResponseDto response = TrainingJobResponseDto.fromEntity(savedJob);
        response.setExecutionMode(isLocal ? "local" : "remote");
        response.setLocalJobSpec(localJobSpec);
        response.setMessage(isLocal
                ? "Run this job with the local agent on your machine, then sync results."
                : "Remote training queued on server.");

        if (!isLocal) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    trainingExecutor.execute(() -> executeRemoteTraining(jobId, runId, dataset.getFilePath(), config));
                }
            });
        }

        return response;
    }

    public void executeRemoteTraining(String jobId, String runId, String datasetPath, TrainingConfigDto config) {
        logger.info("Starting background training for job {}", jobId);
        try {
            updateJobStatus(jobId, runId, "running", 10, null);

            Map<String, Object> payload = new HashMap<>();
            payload.put("job_id", jobId);
            payload.put("dataset_path", datasetPath);
            payload.put("target_column", config.getTargetColumn());
            payload.put("task_type", config.getTaskType());
            payload.put("test_size", config.getTestSize());
            payload.put("cv_folds", config.getCvFolds());
            payload.put("models_to_train", config.getModelsToTrain());
            payload.put("model_hyperparams", config.getModelHyperparams());
            payload.put("enable_tuning", config.getEnableTuning());
            payload.put("tuning_trials", config.getTuningTrials());
            payload.put("tuning_time_budget_sec", config.getTuningTimeBudgetSec());

            JsonNode results = mlEngineClient.train(payload);

            String detectedTaskType = results.has("task_type") ? results.get("task_type").asText() : config.getTaskType();
            JsonNode modelsNode = results.get("models");

            double bestScore = -1e9;
            String bestModelName = null;
            int successfulModels = 0;

            if (modelsNode != null && modelsNode.isArray()) {
                for (JsonNode m : modelsNode) {
                    if (m.has("error")) continue;

                    TrainedModelEntity model = new TrainedModelEntity();
                    model.setJobId(jobId);
                    model.setModelName(m.has("model_name") ? m.get("model_name").asText() : "Unknown");
                    model.setModelType(m.has("model_type") ? m.get("model_type").asText() : null);
                    model.setTaskType(detectedTaskType);
                    model.setFilePath(m.has("file_path") ? m.get("file_path").asText() : null);
                    model.setAccuracy(m.hasNonNull("accuracy") ? m.get("accuracy").asDouble() : null);
                    model.setF1Score(m.hasNonNull("f1_score") ? m.get("f1_score").asDouble() : null);
                    model.setRocAuc(m.hasNonNull("roc_auc") ? m.get("roc_auc").asDouble() : null);
                    model.setRmse(m.hasNonNull("rmse") ? m.get("rmse").asDouble() : null);
                    model.setMae(m.hasNonNull("mae") ? m.get("mae").asDouble() : null);
                    model.setR2Score(m.hasNonNull("r2_score") ? m.get("r2_score").asDouble() : null);
                    model.setMetrics(m.get("metrics"));
                    model.setFeatureImportance(m.get("feature_importance"));
                    model.setConfusionMatrix(m.get("confusion_matrix"));
                    model.setRocCurveData(m.get("roc_curve_data"));
                    model.setCvScores(m.get("cv_scores"));
                    model.setTrainingTime(m.hasNonNull("training_time") ? m.get("training_time").asDouble() : null);

                    trainedModelRepository.save(model);
                    successfulModels++;

                    double score = "classification".equalsIgnoreCase(detectedTaskType)
                            ? (model.getAccuracy() != null ? model.getAccuracy() : -1e9)
                            : (model.getR2Score() != null ? model.getR2Score() : -1e9);

                    if (score > bestScore) {
                        bestScore = score;
                        bestModelName = model.getModelName();
                    }
                }
            }

            if (bestModelName != null) {
                final String champ = bestModelName;
                trainedModelRepository.findAll().stream()
                        .filter(mod -> champ.equals(mod.getModelName()))
                        .findFirst()
                        .ifPresent(mod -> {
                            mod.setIsDeployed(true);
                            trainedModelRepository.save(mod);
                        });
            }

            // Mark completed
            completeJob(jobId, runId, detectedTaskType, bestModelName, bestScore > -1e9 ? bestScore : null, successfulModels);
            logger.info("Training completed for job {} with {} successful models", jobId, successfulModels);

        } catch (Exception e) {
            logger.error("Training failed for job {}: {}", jobId, e.getMessage(), e);
            updateJobStatus(jobId, runId, "failed", 0, e.getMessage());
        }
    }

    @Transactional
    public void updateJobStatus(String jobId, String runId, String status, int progress, String errorMessage) {
        trainingJobRepository.findByJobId(jobId).ifPresent(job -> {
            job.setStatus(status);
            job.setProgress(progress);
            job.setErrorMessage(errorMessage);
            trainingJobRepository.save(job);
        });

        experimentRunRepository.findByRunId(runId).ifPresent(run -> {
            run.setStatus(status);
            run.setErrorMessage(errorMessage);
            if ("failed".equals(status)) {
                run.setCompletedAt(OffsetDateTime.now());
            }
            experimentRunRepository.save(run);
        });
    }

    @Transactional
    public void completeJob(String jobId, String runId, String taskType, String bestModelName, Double bestScore, int modelsCount) {
        trainingJobRepository.findByJobId(jobId).ifPresent(job -> {
            job.setStatus("completed");
            job.setProgress(100);
            job.setTaskType(taskType);
            job.setCompletedAt(OffsetDateTime.now());
            trainingJobRepository.save(job);
        });

        experimentRunRepository.findByRunId(runId).ifPresent(run -> {
            run.setStatus("completed");
            run.setTaskType(taskType);
            run.setBestModelName(bestModelName);
            run.setBestScore(bestScore);
            run.setCompletedAt(OffsetDateTime.now());

            Map<String, Object> summary = Map.of(
                    "best_model", bestModelName != null ? bestModelName : "",
                    "best_score", bestScore != null ? bestScore : 0.0,
                    "models_trained", modelsCount
            );
            run.setSummaryMetrics(objectMapper.valueToTree(summary));
            experimentRunRepository.save(run);
        });
    }

    public TrainingJobResponseDto getStatus(String jobId) {
        TrainingJobEntity job = trainingJobRepository.findByJobId(jobId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Training job not found"));
        TrainingJobResponseDto dto = TrainingJobResponseDto.fromEntity(job);
        experimentRunRepository.findByJobId(jobId).ifPresent(run -> {
            dto.setBestModel(run.getBestModelName());
            dto.setBestScore(run.getBestScore());
        });
        return dto;
    }

    public List<TrainingJobResponseDto> listJobs() {
        return trainingJobRepository.findAllByOrderByCreatedAtDesc()
                .stream()
                .map(TrainingJobResponseDto::fromEntity)
                .toList();
    }

    @Transactional
    public Map<String, Object> syncLocalResults(Map<String, Object> payload) {
        String jobId = (String) payload.get("job_id");
        TrainingJobEntity job = trainingJobRepository.findByJobId(jobId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Training job not found"));

        String status = (String) payload.getOrDefault("status", "completed");
        String taskType = (String) payload.get("task_type");

        trainedModelRepository.deleteByJobId(jobId);

        List<?> models = (List<?>) payload.get("models");
        int count = 0;
        if (models != null) {
            for (Object obj : models) {
                if (obj instanceof Map<?, ?> m) {
                    TrainedModelEntity model = new TrainedModelEntity();
                    model.setJobId(jobId);
                    Object nameObj = m.get("model_name");
                    model.setModelName(nameObj != null ? nameObj.toString() : "Local Model");
                    Object typeObj = m.get("model_type");
                    model.setModelType(typeObj != null ? typeObj.toString() : "local_torch");
                    model.setTaskType(taskType != null ? taskType : job.getTaskType());

                    Map<?, ?> artifactRef = (Map<?, ?>) m.get("artifact_ref");
                    if (artifactRef != null) {
                        model.setFilePath((String) artifactRef.get("path"));
                    }

                    Map<?, ?> metrics = (Map<?, ?>) m.get("metrics");
                    if (metrics != null) {
                        model.setMetrics(objectMapper.valueToTree(metrics));
                        if (metrics.containsKey("accuracy")) model.setAccuracy(((Number) metrics.get("accuracy")).doubleValue());
                        if (metrics.containsKey("f1_score")) model.setF1Score(((Number) metrics.get("f1_score")).doubleValue());
                        if (metrics.containsKey("r2")) model.setR2Score(((Number) metrics.get("r2")).doubleValue());
                    }

                    if (m.get("training_time") instanceof Number t) {
                        model.setTrainingTime(t.doubleValue());
                    }

                    trainedModelRepository.save(model);
                    count++;
                }
            }
        }

        job.setStatus(status);
        job.setProgress("completed".equalsIgnoreCase(status) ? 100 : job.getProgress());
        if (taskType != null) job.setTaskType(taskType);
        job.setCompletedAt(OffsetDateTime.now());
        trainingJobRepository.save(job);

        return Map.of(
                "message", "Local training results synced successfully",
                "job_id", jobId,
                "models_received", count
        );
    }
}
