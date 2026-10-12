package com.axiomcloud.controller;

import com.axiomcloud.dto.PredictionRequestDto;
import com.axiomcloud.dto.PredictionResponseDto;
import com.axiomcloud.service.ModelService;
import com.axiomcloud.service.PredictionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class PredictionController {

    private final PredictionService predictionService;
    private final ModelService modelService;

    public PredictionController(PredictionService predictionService, ModelService modelService) {
        this.predictionService = predictionService;
        this.modelService = modelService;
    }

    @PostMapping("/predict")
    public ResponseEntity<PredictionResponseDto> predict(@RequestBody PredictionRequestDto request) {
        return ResponseEntity.ok(predictionService.predict(request));
    }

    @GetMapping("/deployed-models")
    public ResponseEntity<List<Map<String, Object>>> listDeployedModels() {
        return ResponseEntity.ok(modelService.listDeployedModels());
    }
}
