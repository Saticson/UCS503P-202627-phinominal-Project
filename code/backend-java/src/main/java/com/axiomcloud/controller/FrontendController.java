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
    private final com.axiomcloud.service.MarketplaceService marketplaceService;
    private final com.axiomcloud.service.CommissionService commissionService;

    public FrontendController(DatasetService datasetService,
                              TrainingService trainingService,
                              ModelService modelService,
                              PredictionService predictionService,
                              ObjectMapper objectMapper,
                              com.axiomcloud.service.MarketplaceService marketplaceService,
                              com.axiomcloud.service.CommissionService commissionService) {
        this.datasetService = datasetService;
        this.trainingService = trainingService;
        this.modelService = modelService;
        this.predictionService = predictionService;
        this.objectMapper = objectMapper;
        this.marketplaceService = marketplaceService;
        this.commissionService = commissionService;
    }

    @GetMapping({"/", "/index.html"})
    public String index(Model model) {
        List<DatasetResponseDto> datasets = datasetService.listDatasets();
        List<ModelResponseDto> models = modelService.listModels(null);
        model.addAttribute("datasetCount", datasets.size());
        model.addAttribute("modelCount", models.size());
        model.addAttribute("marketplaceCount", marketplaceService.count());
        model.addAttribute("commissionCount", commissionService.count());
        model.addAttribute("datasets", datasets);
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
        model.addAttribute("marketplaceCount", marketplaceService.count());
        model.addAttribute("commissionCount", commissionService.count());
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

    // ==========================================
    // COMPUTER VISION PREDICTION SANDBOX
    // ==========================================
    @PostMapping("/predict/vision")
    public String runVisionPrediction(
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestParam(value = "sample_image", required = false) String sampleImage,
            @RequestParam(value = "vision_model", required = false) String visionModel,
            Model model) {
        com.axiomcloud.dto.VisionPredictionResponseDto result = 
                predictionService.predictVision(image, sampleImage, visionModel);
        model.addAttribute("visionPrediction", result);
        return "predict :: visionPredictionResult";
    }

    // ==========================================
    // MARKETPLACE (BUY FUNNEL)
    // ==========================================
    @GetMapping("/marketplace")
    public String marketplace(
            @RequestParam(value = "modality", required = false, defaultValue = "ALL") String modality,
            @RequestParam(value = "query", required = false) String query,
            @RequestParam(value = "verified", required = false) Boolean verified,
            @RequestParam(value = "recommend", required = false, defaultValue = "false") boolean recommend,
            @RequestParam(value = "task_type", required = false) String taskType,
            Model model) {

        List<com.axiomcloud.dto.MarketplaceListingDto> listings = 
                marketplaceService.listListings(modality, query, verified);
        model.addAttribute("listings", listings);
        model.addAttribute("activeModality", modality);
        model.addAttribute("activeQuery", query != null ? query : "");
        model.addAttribute("verifiedOnly", verified != null && verified);
        model.addAttribute("isRecommended", recommend);

        if (recommend) {
            String recModality = (modality != null && !modality.equalsIgnoreCase("ALL")) ? modality : "TABULAR";
            List<com.axiomcloud.dto.MarketplaceListingDto> recommended = 
                    marketplaceService.recommendListings(recModality, taskType);
            model.addAttribute("recommendedListings", recommended);
        }

        return "marketplace";
    }

    @GetMapping("/marketplace/{id}")
    public String listingDetail(@PathVariable("id") String id, Model model) {
        com.axiomcloud.dto.MarketplaceListingDto listing = marketplaceService.getListing(id)
                .orElseThrow(() -> new org.springframework.web.server.ResponseStatusException(
                        org.springframework.http.HttpStatus.NOT_FOUND, "Marketplace listing not found"));
        model.addAttribute("listing", listing);
        return "listing-detail";
    }

    @PostMapping("/marketplace/publish")
    public String publishListing(
            @ModelAttribute com.axiomcloud.dto.MarketplaceListingDto dto,
            Model model) {
        com.axiomcloud.dto.MarketplaceListingDto created = marketplaceService.publish(dto);
        return "redirect:/marketplace/" + created.getId();
    }

    // ==========================================
    // HIRE FUNNEL (EXPERT COMMISSIONS)
    // ==========================================
    @GetMapping("/hire")
    public String hire(Model model) {
        model.addAttribute("commissions", commissionService.listCommissions());
        return "hire";
    }

    @PostMapping("/hire/submit")
    public String submitCommission(
            @RequestParam("client_name") String clientName,
            @RequestParam("email") String email,
            @RequestParam("title") String title,
            @RequestParam("modality") String modality,
            @RequestParam("task_type") String taskType,
            @RequestParam("budget") String budget,
            @RequestParam("description") String description,
            Model model) {
        com.axiomcloud.dto.ExpertCommissionDto commission = new com.axiomcloud.dto.ExpertCommissionDto(
                null, clientName, email, title, modality, taskType, budget, description, "OPEN"
        );
        commissionService.add(commission);
        model.addAttribute("commissions", commissionService.listCommissions());
        model.addAttribute("submitSuccess", true);
        return "redirect:/hire";
    }
}

