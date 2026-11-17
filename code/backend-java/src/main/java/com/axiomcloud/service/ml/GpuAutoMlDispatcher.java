package com.axiomcloud.service.ml;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;
import java.util.Map;

/**
 * Orchestrates training jobs between the optional Linux/WSL2 Python RAPIDS (cuML/cuDF) worker
 * and the embedded multi-threaded Smile CPU engine.
 * Records the exact backend used, hardware specs, and transparent fallback reasons.
 */
@Component
public class GpuAutoMlDispatcher {

    private static final Logger logger = LoggerFactory.getLogger(GpuAutoMlDispatcher.class);

    private final GpuEnvironmentDetector gpuDetector;
    private final SmileTrainer smileTrainer;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final String pythonWorkerUrl;

    public GpuAutoMlDispatcher(GpuEnvironmentDetector gpuDetector,
                               SmileTrainer smileTrainer,
                               RestTemplateBuilder restTemplateBuilder,
                               ObjectMapper objectMapper,
                               @Value("${axiomcloud.ml-engine.url:http://localhost:8001}") String pythonWorkerUrl) {
        this.gpuDetector = gpuDetector;
        this.smileTrainer = smileTrainer;
        this.objectMapper = objectMapper;
        this.pythonWorkerUrl = pythonWorkerUrl.replaceAll("/+$", "");
        this.restTemplate = restTemplateBuilder
                .setConnectTimeout(Duration.ofMillis(800))
                .setReadTimeout(Duration.ofMinutes(10))
                .build();
    }

    public JsonNode dispatchTraining(Map<String, Object> payload) {
        // Attempt Python GPU worker if hardware and worker are configured
        if (gpuDetector.isGpuAccelerated()) {
            try {
                logger.info("Dispatching tabular training to Python RAPIDS GPU worker at {}", pythonWorkerUrl);
                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);
                HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
                ResponseEntity<String> response = restTemplate.postForEntity(
                        pythonWorkerUrl + "/internal/train", entity, String.class);

                if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                    JsonNode node = objectMapper.readTree(response.getBody());
                    logger.info("Python RAPIDS worker successfully completed training job.");
                    return node;
                }
            } catch (Exception e) {
                logger.warn("Python GPU worker dispatch failed ({}); falling back to CPU Smile engine with audit log.", e.getMessage());
            }
        }

        // Fallback to embedded Smile CPU engine with honest metadata annotation
        logger.info("Executing on embedded multi-threaded Smile CPU engine. Fallback reason: {}",
                gpuDetector.getFallbackReason());

        ObjectNode cpuResult = smileTrainer.train(payload);

        // Annotate result with execution reality
        cpuResult.put("backend_used", "EMBEDDED_SMILE_CPU");
        cpuResult.put("hardware_gpu_detected", gpuDetector.isHardwareGpuPresent());
        cpuResult.put("gpu_device_name", gpuDetector.getDeviceName());
        cpuResult.put("fallback_reason", gpuDetector.getFallbackReason());

        JsonNode modelsNode = cpuResult.get("models");
        if (modelsNode != null && modelsNode.isArray()) {
            for (JsonNode m : modelsNode) {
                if (m instanceof ObjectNode obj) {
                    obj.put("backend_used", "EMBEDDED_SMILE_CPU");
                    obj.put("fallback_reason", gpuDetector.getFallbackReason());
                    JsonNode metrics = obj.get("metrics");
                    if (metrics instanceof ObjectNode metObj) {
                        metObj.put("backend_used", "EMBEDDED_SMILE_CPU");
                        metObj.put("gpu_accelerated", false);
                        metObj.put("fallback_reason", gpuDetector.getFallbackReason());
                    }
                }
            }
        }

        return cpuResult;
    }
}
