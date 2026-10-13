package com.axiomcloud.controller;

import com.axiomcloud.dto.DatasetResponseDto;
import com.axiomcloud.service.DatasetService;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class DatasetController {

    private final DatasetService datasetService;

    public DatasetController(DatasetService datasetService) {
        this.datasetService = datasetService;
    }

    @PostMapping(value = {"/upload-dataset", "/datasets/upload"}, consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<DatasetResponseDto> uploadDataset(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "name", required = false) String name,
            @RequestParam(value = "target_column", required = false) String targetColumn) {
        return ResponseEntity.ok(datasetService.uploadDataset(file, name, targetColumn));
    }

    @GetMapping({"/datasets", "/datasets/list"})
    public ResponseEntity<List<DatasetResponseDto>> listDatasets() {
        return ResponseEntity.ok(datasetService.listDatasets());
    }

    @GetMapping("/datasets/{id}")
    public ResponseEntity<DatasetResponseDto> getDataset(@PathVariable("id") int id) {
        return ResponseEntity.ok(datasetService.getDatasetDto(id));
    }

    @DeleteMapping("/datasets/{id}")
    public ResponseEntity<Map<String, String>> deleteDataset(@PathVariable("id") int id) {
        datasetService.deleteDataset(id);
        return ResponseEntity.ok(Map.of("message", "Dataset deleted successfully"));
    }

    @GetMapping("/datasets/{id}/quality-report")
    public ResponseEntity<JsonNode> getQualityReport(@PathVariable("id") int id) {
        return ResponseEntity.ok(datasetService.getQualityReport(id));
    }

    @GetMapping("/datasets/{id}/clean-preview")
    public ResponseEntity<JsonNode> getCleanPreview(
            @PathVariable("id") int id,
            @RequestParam(value = "preview_rows", defaultValue = "10") int previewRows) {
        return ResponseEntity.ok(datasetService.getCleanPreview(id, previewRows));
    }

    @PostMapping("/datasets/{id}/clean-and-save")
    public ResponseEntity<DatasetResponseDto> cleanAndSave(
            @PathVariable("id") int id,
            @RequestParam(value = "name_suffix", defaultValue = "cleaned") String nameSuffix) {
        return ResponseEntity.ok(datasetService.cleanAndSave(id, nameSuffix));
    }

    @GetMapping("/datasets/{id}/profile")
    public ResponseEntity<JsonNode> getProfile(@PathVariable("id") int id) {
        return ResponseEntity.ok(datasetService.getProfile(id));
    }

    @GetMapping("/datasets/{id}/leakage-report")
    public ResponseEntity<JsonNode> getLeakageReport(@PathVariable("id") int id) {
        return ResponseEntity.ok(datasetService.getLeakageReport(id));
    }

    @GetMapping("/datasets/{id}/drift-baseline")
    public ResponseEntity<JsonNode> getDriftBaseline(@PathVariable("id") int id) {
        return ResponseEntity.ok(datasetService.getDriftBaseline(id));
    }

    @GetMapping("/datasets/{id}/eda-report")
    public ResponseEntity<JsonNode> getEdaReport(@PathVariable("id") int id) {
        return ResponseEntity.ok(datasetService.getEdaReport(id));
    }

    @GetMapping("/datasets/{id}/analytics-report")
    public ResponseEntity<JsonNode> getAnalyticsReport(
            @PathVariable("id") int id,
            @RequestParam(value = "target_column", required = false) String targetColumn,
            @RequestParam(value = "model_type", required = false) String modelType,
            @RequestParam(value = "model_id", required = false) Integer modelId) {
        return ResponseEntity.ok(datasetService.getAnalyticsReport(id, targetColumn, modelType, modelId));
    }

    @PostMapping("/load-example/{dataset_key}")
    public ResponseEntity<DatasetResponseDto> loadExample(@PathVariable("dataset_key") String datasetKey) {
        return ResponseEntity.ok(datasetService.loadExampleDataset(datasetKey));
    }

    @GetMapping("/example-datasets")
    public ResponseEntity<List<Map<String, String>>> listExampleDatasets() {
        List<Map<String, String>> examples = List.of(
                Map.of(
                        "key", "iris",
                        "name", "Iris Species",
                        "description", "Classic 3-class flower classification dataset",
                        "target", "species",
                        "task_type", "classification"
                ),
                Map.of(
                        "key", "california_housing",
                        "name", "California Housing",
                        "description", "Median house value prediction across California districts",
                        "target", "MedHouseVal",
                        "task_type", "regression"
                ),
                Map.of(
                        "key", "titanic",
                        "name", "Titanic Survival",
                        "description", "Passenger survival classification dataset",
                        "target", "survived",
                        "task_type", "classification"
                ),
                Map.of(
                        "key", "diabetes",
                        "name", "Diabetes Progression",
                        "description", "Quantitative disease progression measure one year after baseline",
                        "target", "target",
                        "task_type", "regression"
                )
        );
        return ResponseEntity.ok(examples);
    }
}
