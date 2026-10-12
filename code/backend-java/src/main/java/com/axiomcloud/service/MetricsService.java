package com.axiomcloud.service;

import com.axiomcloud.dto.DashboardSummaryDto;
import com.axiomcloud.dto.ModelResponseDto;
import com.axiomcloud.model.TrainedModelEntity;
import com.axiomcloud.model.TrainingJobEntity;
import com.axiomcloud.repository.DatasetRepository;
import com.axiomcloud.repository.TrainedModelRepository;
import com.axiomcloud.repository.TrainingJobRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.*;

@Service
public class MetricsService {

    private final DatasetRepository datasetRepository;
    private final TrainedModelRepository trainedModelRepository;
    private final TrainingJobRepository trainingJobRepository;

    public MetricsService(DatasetRepository datasetRepository,
                          TrainedModelRepository trainedModelRepository,
                          TrainingJobRepository trainingJobRepository) {
        this.datasetRepository = datasetRepository;
        this.trainedModelRepository = trainedModelRepository;
        this.trainingJobRepository = trainingJobRepository;
    }

    public DashboardSummaryDto getDashboardSummary() {
        long totalDatasets = datasetRepository.count();
        long totalModels = trainedModelRepository.count();
        long deployedModels = trainedModelRepository.countByIsDeployedTrue();
        long completedJobs = trainingJobRepository.countByStatus("completed");
        long failedJobs = trainingJobRepository.countByStatus("failed");

        return new DashboardSummaryDto(totalDatasets, totalModels, deployedModels, completedJobs, failedJobs);
    }

    public Map<String, Object> getJobMetrics(String jobId) {
        TrainingJobEntity job = trainingJobRepository.findByJobId(jobId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Training job not found"));

        List<TrainedModelEntity> models = trainedModelRepository.findByJobIdOrderByCreatedAtDesc(jobId);
        String taskType = job.getTaskType() != null ? job.getTaskType() : "classification";

        List<ModelResponseDto> modelDtos = models.stream()
                .map(ModelResponseDto::fromEntity)
                .toList();

        // Sort leaderboard
        String primaryMetric = "classification".equalsIgnoreCase(taskType) ? "accuracy" : "r2_score";
        List<Map<String, Object>> leaderboard = new ArrayList<>();

        for (int rank = 0; rank < models.size(); rank++) {
            TrainedModelEntity m = models.get(rank);
            Map<String, Object> entry = new HashMap<>();
            entry.put("rank", rank + 1);
            entry.put("id", m.getId());
            entry.put("model_name", m.getModelName());
            entry.put("model_type", m.getModelType());
            entry.put("accuracy", m.getAccuracy());
            entry.put("f1_score", m.getF1Score());
            entry.put("r2_score", m.getR2Score());
            entry.put("rmse", m.getRmse());
            entry.put("is_deployed", m.getIsDeployed());
            leaderboard.add(entry);
        }

        Map<String, Object> result = new HashMap<>();
        result.put("job_id", jobId);
        result.put("task_type", taskType);
        result.put("status", job.getStatus());
        result.put("models", modelDtos);
        result.put("leaderboard", leaderboard);
        result.put("metric_catalog", List.of("accuracy", "f1_score", "roc_auc", "rmse", "mae", "r2_score"));
        result.put("chart_data", Map.of());

        return result;
    }
}
