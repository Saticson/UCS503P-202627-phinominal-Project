package com.axiomcloud.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

@Service
public class MlEngineClient {

    private static final Logger logger = LoggerFactory.getLogger(MlEngineClient.class);

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final String mlEngineUrl;

    public MlEngineClient(@Value("${axiomcloud.ml-engine.url:http://localhost:8001}") String mlEngineUrl,
                          RestTemplateBuilder restTemplateBuilder,
                          ObjectMapper objectMapper) {
        this.mlEngineUrl = mlEngineUrl.replaceAll("/+$", "");
        this.objectMapper = objectMapper;
        this.restTemplate = restTemplateBuilder
                .setConnectTimeout(Duration.ofSeconds(10))
                .setReadTimeout(Duration.ofMinutes(30))
                .build();
    }

    public JsonNode train(Map<String, Object> requestPayload) {
        String url = mlEngineUrl + "/internal/train";
        return postForJson(url, requestPayload);
    }

    public JsonNode predict(Map<String, Object> requestPayload) {
        String url = mlEngineUrl + "/internal/predict";
        return postForJson(url, requestPayload);
    }

    public JsonNode explainShap(Map<String, Object> payload) {
        return postForJson(mlEngineUrl + "/internal/models/explain/shap", payload);
    }

    public JsonNode explainLime(Map<String, Object> payload) {
        return postForJson(mlEngineUrl + "/internal/models/explain/lime", payload);
    }

    public JsonNode datasetProfile(int datasetId, String datasetName, String filePath, String targetColumn) {
        String url = mlEngineUrl + "/internal/datasets/profile";
        Map<String, Object> body = new HashMap<>();
        body.put("dataset_id", datasetId);
        body.put("dataset_name", datasetName);
        body.put("file_path", filePath);
        body.put("target_column", targetColumn);
        return postForJson(url, body);
    }

    public JsonNode qualityReport(String filePath) {
        String url = mlEngineUrl + "/internal/datasets/quality-report";
        Map<String, Object> body = Map.of("file_path", filePath);
        return postForJson(url, body);
    }

    public JsonNode cleanPreview(String filePath, int previewRows) {
        return postForJson(mlEngineUrl + "/internal/datasets/clean-preview",
                Map.of("file_path", filePath, "preview_rows", previewRows));
    }

    public JsonNode cleanAndSave(String filePath, String outputFilePath) {
        return postForJson(mlEngineUrl + "/internal/datasets/clean-and-save",
                Map.of("file_path", filePath, "output_file_path", outputFilePath));
    }

    public JsonNode leakageReport(int datasetId, String datasetName, String filePath, String targetColumn) {
        Map<String, Object> body = new HashMap<>();
        body.put("dataset_id", datasetId);
        body.put("dataset_name", datasetName);
        body.put("file_path", filePath);
        body.put("target_column", targetColumn);
        return postForJson(mlEngineUrl + "/internal/datasets/leakage-report", body);
    }

    public JsonNode driftBaseline(String filePath, String targetColumn) {
        Map<String, Object> body = new HashMap<>();
        body.put("file_path", filePath);
        body.put("target_column", targetColumn);
        return postForJson(mlEngineUrl + "/internal/datasets/drift-baseline", body);
    }

    public JsonNode monitoring(String trainingFilePath, String comparisonFilePath, String targetColumn) {
        Map<String, Object> body = new HashMap<>();
        body.put("training_file_path", trainingFilePath);
        body.put("comparison_file_path", comparisonFilePath);
        body.put("target_column", targetColumn);
        return postForJson(mlEngineUrl + "/internal/models/monitoring", body);
    }

    public JsonNode edaReport(String datasetName, String filePath, String targetColumn) {
        String url = mlEngineUrl + "/internal/datasets/eda-report";
        Map<String, Object> body = new HashMap<>();
        body.put("dataset_name", datasetName);
        body.put("file_path", filePath);
        body.put("target_column", targetColumn);
        return postForJson(url, body);
    }

    public JsonNode analyticsReport(int datasetId, String datasetName, String filePath, String targetColumn, Integer modelId, String modelType) {
        String url = mlEngineUrl + "/internal/datasets/analytics-report";
        Map<String, Object> body = new HashMap<>();
        body.put("dataset_id", datasetId);
        body.put("dataset_name", datasetName);
        body.put("file_path", filePath);
        body.put("target_column", targetColumn);
        if (modelId != null) body.put("model_id", modelId);
        if (modelType != null) body.put("model_type", modelType);
        return postForJson(url, body);
    }

    private JsonNode postForJson(String url, Object body) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Object> entity = new HttpEntity<>(body, headers);
            ResponseEntity<String> response = restTemplate.postForEntity(url, entity, String.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return objectMapper.readTree(response.getBody());
            }
            throw new RuntimeException("ML Engine returned non-2xx status: " + response.getStatusCode());
        } catch (Exception e) {
            logger.error("Error communicating with ML engine at {}: {}", url, e.getMessage());
            throw new RuntimeException("Failed to call ML engine: " + e.getMessage(), e);
        }
    }

    private JsonNode getForJson(String url) {
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return objectMapper.readTree(response.getBody());
            }
            throw new RuntimeException("ML Engine returned non-2xx status: " + response.getStatusCode());
        } catch (Exception e) {
            logger.error("Error calling ML engine GET {}: {}", url, e.getMessage());
            throw new RuntimeException("Failed to call ML engine: " + e.getMessage(), e);
        }
    }
}
