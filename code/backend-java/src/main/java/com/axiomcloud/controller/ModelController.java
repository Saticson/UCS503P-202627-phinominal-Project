package com.axiomcloud.controller;

import com.axiomcloud.dto.ModelResponseDto;
import com.axiomcloud.service.ModelService;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class ModelController {

    private final ModelService modelService;

    public ModelController(ModelService modelService) {
        this.modelService = modelService;
    }

    @GetMapping("/models")
    public ResponseEntity<List<ModelResponseDto>> listModels(
            @RequestParam(value = "job_id", required = false) String jobId) {
        return ResponseEntity.ok(modelService.listModels(jobId));
    }

    @GetMapping("/models/{id}")
    public ResponseEntity<ModelResponseDto> getModel(@PathVariable("id") int id) {
        return ResponseEntity.ok(modelService.getModel(id));
    }

    @GetMapping("/models/{id}/inference-template")
    public ResponseEntity<Map<String, Object>> getInferenceTemplate(
            @PathVariable("id") int id,
            @RequestParam(value = "randomize", defaultValue = "true") boolean randomize) {
        return ResponseEntity.ok(modelService.getInferenceTemplate(id, randomize));
    }

    @GetMapping("/models/{id}/shap")
    public ResponseEntity<JsonNode> getShap(
            @PathVariable("id") int id,
            @RequestParam(value = "sample_index", defaultValue = "0") int sampleIndex,
            @RequestParam(value = "nsamples", defaultValue = "200") int nsamples) {
        return ResponseEntity.ok(modelService.getShapExplanation(id, sampleIndex, nsamples));
    }

    @PostMapping("/models/{id}/lime")
    public ResponseEntity<JsonNode> getLimePost(
            @PathVariable("id") int id,
            @RequestBody(required = false) Map<String, Object> body) {
        int sampleIndex = body != null && body.containsKey("sample_index") ? ((Number) body.get("sample_index")).intValue() : 0;
        int numFeatures = body != null && body.containsKey("num_features") ? ((Number) body.get("num_features")).intValue() : 12;
        Map<String, Object> customInput = body != null && body.get("custom_input") instanceof Map<?, ?> m
                ? (Map<String, Object>) m : null;
        return ResponseEntity.ok(modelService.getLimeExplanation(id, sampleIndex, numFeatures, customInput));
    }

    @GetMapping("/models/{id}/lime")
    public ResponseEntity<JsonNode> getLimeGet(
            @PathVariable("id") int id,
            @RequestParam(value = "sample_index", defaultValue = "0") int sampleIndex,
            @RequestParam(value = "num_features", defaultValue = "12") int numFeatures) {
        return ResponseEntity.ok(modelService.getLimeExplanation(id, sampleIndex, numFeatures, null));
    }

    @GetMapping("/models/{id}/monitoring")
    public ResponseEntity<Map<String, Object>> getMonitoring(
            @PathVariable("id") int id,
            @RequestParam(value = "compare_dataset_id", required = false) Integer compareDatasetId) {
        return ResponseEntity.ok(modelService.getMonitoring(id, compareDatasetId));
    }

    @PostMapping("/models/{id}/deploy")
    public ResponseEntity<Map<String, Object>> deployModel(@PathVariable("id") int id) {
        return ResponseEntity.ok(modelService.deployModel(id));
    }

    @PostMapping("/models/{id}/undeploy")
    public ResponseEntity<Map<String, Object>> undeployModel(@PathVariable("id") int id) {
        return ResponseEntity.ok(modelService.undeployModel(id));
    }

    @GetMapping("/models/{id}/download")
    public ResponseEntity<Resource> downloadModel(@PathVariable("id") int id) {
        Resource file = modelService.downloadModelFile(id);
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"model_" + id + ".joblib\"")
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(file);
    }

    @DeleteMapping("/models/{id}")
    public ResponseEntity<Map<String, String>> deleteModel(@PathVariable("id") int id) {
        modelService.deleteModel(id);
        return ResponseEntity.ok(Map.of("message", "Model deleted successfully"));
    }
}
