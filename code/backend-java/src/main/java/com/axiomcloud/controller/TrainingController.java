package com.axiomcloud.controller;

import com.axiomcloud.dto.ExperimentRunResponseDto;
import com.axiomcloud.dto.TrainingConfigDto;
import com.axiomcloud.dto.TrainingJobResponseDto;
import com.axiomcloud.model.TrainingJobEntity;
import com.axiomcloud.repository.ExperimentRunRepository;
import com.axiomcloud.repository.TrainingJobRepository;
import com.axiomcloud.service.TrainingService;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;
import org.springframework.web.server.ResponseStatusException;

import java.io.File;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class TrainingController {

    private final TrainingService trainingService;
    private final TrainingJobRepository trainingJobRepository;
    private final ExperimentRunRepository experimentRunRepository;

    public TrainingController(TrainingService trainingService,
                              TrainingJobRepository trainingJobRepository,
                              ExperimentRunRepository experimentRunRepository) {
        this.trainingService = trainingService;
        this.trainingJobRepository = trainingJobRepository;
        this.experimentRunRepository = experimentRunRepository;
    }

    @PostMapping({"/train-model", "/training/launch"})
    public ResponseEntity<TrainingJobResponseDto> trainModel(@Valid @RequestBody TrainingConfigDto config) {
        return ResponseEntity.ok(trainingService.launchTraining(config));
    }

    @GetMapping("/training-status/{job_id}")
    public ResponseEntity<TrainingJobResponseDto> getStatus(@PathVariable("job_id") String jobId) {
        return ResponseEntity.ok(trainingService.getStatus(jobId));
    }

    @GetMapping("/training-jobs")
    public ResponseEntity<List<TrainingJobResponseDto>> listJobs() {
        return ResponseEntity.ok(trainingService.listJobs());
    }

    @GetMapping("/training/local-job-spec/{job_id}")
    public ResponseEntity<JsonNode> getLocalJobSpec(@PathVariable("job_id") String jobId) {
        TrainingJobEntity job = trainingJobRepository.findByJobId(jobId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Job not found"));

        if (job.getConfig() != null && job.getConfig().has("local_job_spec")) {
            return ResponseEntity.ok(job.getConfig().get("local_job_spec"));
        }
        throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Local job spec not found");
    }

    @PostMapping("/training/local-sync")
    public ResponseEntity<Map<String, Object>> syncLocal(@RequestBody Map<String, Object> payload) {
        return ResponseEntity.ok(trainingService.syncLocalResults(payload));
    }

    @GetMapping("/training/model-catalog")
    public ResponseEntity<Map<String, Object>> modelCatalog() {
        List<String> classification = List.of(
                "Logistic Regression", "Linear SVM", "RBF SVM", "KNN", "Decision Tree",
                "Random Forest", "Extra Trees", "AdaBoost", "XGBoost", "LightGBM", "Gradient Boosting",
                "MLP Neural Net", "Deep MLP Neural Net"
        );
        List<String> regression = List.of(
                "Linear Regression", "Linear SVR", "RBF SVR", "KNN", "Decision Tree",
                "Random Forest", "Extra Trees", "AdaBoost", "XGBoost", "LightGBM", "Gradient Boosting",
                "MLP Neural Net", "Deep MLP Neural Net"
        );
        return ResponseEntity.ok(Map.of(
                "classification", classification,
                "regression", regression,
                "all", classification
        ));
    }

    @GetMapping("/training/local-agent/download")
    public ResponseEntity<Resource> downloadLocalAgent() {
        File agentFile = new File("./local_agent.py");
        if (!agentFile.exists()) {
            agentFile = new File("../local_agent.py");
        }
        if (!agentFile.exists()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "local_agent.py not found");
        }
        Resource resource = new FileSystemResource(agentFile);
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"local_agent.py\"")
                .contentType(MediaType.parseMediaType("text/x-python"))
                .body(resource);
    }

    @GetMapping("/experiments")
    public ResponseEntity<List<ExperimentRunResponseDto>> listExperiments(
            @RequestParam(value = "status", required = false) String status) {
        List<ExperimentRunResponseDto> runs = (status != null && !status.isBlank())
                ? experimentRunRepository.findByStatusOrderByStartedAtDesc(status).stream().map(ExperimentRunResponseDto::fromEntity).toList()
                : experimentRunRepository.findAllByOrderByStartedAtDesc().stream().map(ExperimentRunResponseDto::fromEntity).toList();
        return ResponseEntity.ok(runs);
    }

    @GetMapping("/experiments/{run_id}")
    public ResponseEntity<ExperimentRunResponseDto> getExperiment(@PathVariable("run_id") String runId) {
        return experimentRunRepository.findByRunId(runId)
                .map(ExperimentRunResponseDto::fromEntity)
                .map(ResponseEntity::ok)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Experiment run not found"));
    }
}
