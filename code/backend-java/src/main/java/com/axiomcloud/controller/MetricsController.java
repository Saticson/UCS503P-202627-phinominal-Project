package com.axiomcloud.controller;

import com.axiomcloud.dto.DashboardSummaryDto;
import com.axiomcloud.service.MetricsService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class MetricsController {

    private final MetricsService metricsService;

    public MetricsController(MetricsService metricsService) {
        this.metricsService = metricsService;
    }

    @GetMapping("/metrics/{job_id}")
    public ResponseEntity<Map<String, Object>> getMetrics(@PathVariable("job_id") String jobId) {
        return ResponseEntity.ok(metricsService.getJobMetrics(jobId));
    }

    @GetMapping("/dashboard/summary")
    public ResponseEntity<DashboardSummaryDto> getDashboardSummary() {
        return ResponseEntity.ok(metricsService.getDashboardSummary());
    }
}
