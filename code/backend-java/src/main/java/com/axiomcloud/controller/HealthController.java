package com.axiomcloud.controller;

import com.axiomcloud.service.ml.GpuEnvironmentDetector;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
public class HealthController {

    private final GpuEnvironmentDetector gpuDetector;

    public HealthController(GpuEnvironmentDetector gpuDetector) {
        this.gpuDetector = gpuDetector;
    }

    @GetMapping("/api/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> resp = new HashMap<>();
        resp.put("status", "ok");
        resp.put("hardware_gpu_detected", gpuDetector.isHardwareGpuPresent());
        resp.put("gpu_accelerated", gpuDetector.isGpuAccelerated());
        resp.put("gpu_device", gpuDetector.getDeviceName());
        resp.put("vram_mb", gpuDetector.getTotalVramMb());
        resp.put("effective_backend", gpuDetector.getEffectiveBackend());
        resp.put("fallback_reason", gpuDetector.getFallbackReason());
        return ResponseEntity.ok(resp);
    }

    @GetMapping("/api/health/gpu")
    public ResponseEntity<Map<String, Object>> gpuDiagnostics() {
        return ResponseEntity.ok(gpuDetector.getHardwareDiagnostics());
    }

    @GetMapping("/api")
    public Map<String, String> root() {
        return Map.of("message", "Sukio AI API - Java Spring Boot Backend");
    }
}
