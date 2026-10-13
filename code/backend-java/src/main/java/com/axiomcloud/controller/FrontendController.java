package com.axiomcloud.controller;

import com.axiomcloud.dto.DatasetResponseDto;
import com.axiomcloud.dto.ModelResponseDto;
import com.axiomcloud.dto.PredictionRequestDto;
import com.axiomcloud.dto.PredictionResponseDto;
import com.axiomcloud.dto.TrainingConfigDto;
import com.axiomcloud.dto.TrainingJobResponseDto;
import com.axiomcloud.service.DatasetService;
import com.axiomcloud.service.ModelService;
import com.axiomcloud.service.PredictionService;
import com.axiomcloud.service.TrainingService;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

@Controller
public class FrontendController {

    private final DatasetService datasetService;
    private final TrainingService trainingService;
    private final ModelService modelService;
    private final PredictionService predictionService;
    private final ObjectMapper objectMapper;

    public FrontendController(DatasetService datasetService,
                              TrainingService trainingService,
                              ModelService modelService,
                              PredictionService predictionService,
                              ObjectMapper objectMapper) {
        this.datasetService = datasetService;
        this.trainingService = trainingService;
        this.modelService = modelService;
        this.predictionService = predictionService;
        this.objectMapper = objectMapper;
    }

    @GetMapping({"/", "/index.html"})
    public String index() {
        return "index";
    }

    // ==========================================
    // DASHBOARD
    // ==========================================
    @GetMapping("/dashboard")
    public String dashboard(Model model) {
        List<DatasetResponseDto> datasets = datasetService.listDatasets();
        List<ModelResponseDto> models = modelService.listModels(null);
        List<TrainingJobResponseDto> jobs = trainingService.listJobs();

        model.addAttribute("datasetCount", datasets.size());
        model.addAttribute("modelCount", models.size());
        model.addAttribute("jobCount", jobs.size());
        model.addAttribute("recentJobs", jobs);
        model.addAttribute("datasets", datasets);
        return "dashboard";
    }

    // ==========================================
    // DATASETS
    // ==========================================
    @GetMapping("/datasets")
    public String datasets(Model model) {
        model.addAttribute("datasets", datasetService.listDatasets());
        return "datasets";
    }

    @PostMapping("/datasets/load-benchmark/{key}")
    public String loadBenchmark(@PathVariable("key") String key, Model model) {
        datasetService.loadExampleDataset(key);
        model.addAttribute("datasets", datasetService.listDatasets());
        return "datasets :: datasetTableBody";
    }

    @PostMapping(value = "/datasets/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public String uploadDataset(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "name", required = false) String name,
            @RequestParam(value = "target_column", required = false) String targetColumn,
            Model model) {
        datasetService.uploadDataset(file, name, targetColumn);
        model.addAttribute("datasets", datasetService.listDatasets());
        return "datasets :: datasetTableBody";
    }

    @DeleteMapping("/datasets/delete/{id}")
    public String deleteDataset(@PathVariable("id") int id, Model model) {
        datasetService.deleteDataset(id);
        model.addAttribute("datasets", datasetService.listDatasets());
        return "datasets :: datasetTableBody";
    }

    // ==========================================
    // TRAINING STUDIO
    // ==========================================
    @GetMapping("/training")
    public String training(@RequestParam(value = "dataset_id", required = false) Integer selectedDatasetId, Model model) {
        List<DatasetResponseDto> datasets = datasetService.listDatasets();
        model.addAttribute("datasets", datasets);
        model.addAttribute("selectedDatasetId", selectedDatasetId);
        return "training";
    }

    @PostMapping("/training/launch")
    public String launchTraining(
            @RequestParam("dataset_id") int datasetId,
            @RequestParam("target_column") String targetColumn,
            @RequestParam(value = "task_type", defaultValue = "classification") String taskType,
            @RequestParam(value = "models_to_train", required = false) List<String> modelsToTrain,
            @RequestParam(value = "tuning_trials", defaultValue = "10") int tuningTrials,
            @RequestParam(value = "cv_folds", defaultValue = "3") int cvFolds,
            Model model) {
        if (datasetService.listDatasets().isEmpty()) {
            datasetService.loadExampleDataset("titanic");
        }
        TrainingConfigDto config = new TrainingConfigDto();
        config.setDatasetId(datasetId);
        config.setTargetColumn(targetColumn);
        config.setTaskType(taskType);
        config.setModelsToTrain(modelsToTrain != null && !modelsToTrain.isEmpty()
                ? modelsToTrain
                : List.of("Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"));
        config.setTuningTrials(tuningTrials);
        config.setCvFolds(cvFolds);
        config.setEnableTuning(true);

        TrainingJobResponseDto job = trainingService.launchTraining(config);
        model.addAttribute("job", job);
        return "training :: trainingMonitor";
    }

    @GetMapping("/training/status/{jobId}")
    public String getTrainingStatus(@PathVariable("jobId") String jobId, Model model) {
        TrainingJobResponseDto job = trainingService.getStatus(jobId);
        model.addAttribute("job", job);
        return "training :: trainingMonitor";
    }

    // ==========================================
    // MODELS LEADERBOARD
    // ==========================================
    @GetMapping("/models")
    public String models(Model model) {
        model.addAttribute("models", modelService.listModels(null));
        return "models";
    }

    // ==========================================
    // INFERENCE SANDBOX
    // ==========================================
    @GetMapping("/predict")
    public String predict(@RequestParam(value = "model_id", required = false) Integer selectedModelId, Model model) {
        List<ModelResponseDto> models = modelService.listModels(null);
        model.addAttribute("models", models);
        int defaultModelId = selectedModelId != null
                ? selectedModelId
                : (!models.isEmpty() ? models.get(0).getId() : 1);
        model.addAttribute("selectedModelId", defaultModelId);
        return "predict";
    }

    @PostMapping("/predict/run")
    public String runPrediction(
            @RequestParam("model_id") int modelId,
            @RequestParam(value = "features", required = false) String featuresJson,
            Model model) {
        PredictionRequestDto request = new PredictionRequestDto();
        request.setModelId(modelId);

        if (featuresJson != null && !featuresJson.trim().isEmpty()) {
            try {
                Map<String, Object> map = objectMapper.readValue(featuresJson, new TypeReference<>() {});
                request.setFeatures(map);
            } catch (Exception e) {
                request.setFeatures(Map.of("raw_input", featuresJson));
            }
        } else {
            request.setFeatures(Map.of("sample_feature", 1.0));
        }

        PredictionResponseDto response = predictionService.predict(request);
        model.addAttribute("prediction", response);
        return "predict :: predictionResult";
    }
}
