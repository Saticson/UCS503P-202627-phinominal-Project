package com.axiomcloud.service;

import com.axiomcloud.dto.PredictionRequestDto;
import com.axiomcloud.dto.PredictionResponseDto;
import com.axiomcloud.model.TrainedModelEntity;
import com.axiomcloud.repository.TrainedModelRepository;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.HashMap;
import java.util.Map;

@Service
public class PredictionService {

    private final TrainedModelRepository trainedModelRepository;
    private final MlEngineClient mlEngineClient;

    public PredictionService(TrainedModelRepository trainedModelRepository,
                             MlEngineClient mlEngineClient) {
        this.trainedModelRepository = trainedModelRepository;
        this.mlEngineClient = mlEngineClient;
    }

    public PredictionResponseDto predict(PredictionRequestDto request) {
        TrainedModelEntity model = trainedModelRepository.findById(request.getModelId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Model not found"));

        if (model.getFilePath() == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Model file path not available");
        }

        Map<String, Object> payload = new HashMap<>();
        payload.put("model_id", model.getId());
        payload.put("file_path", model.getFilePath());
        payload.put("features", request.getFeatures());
        payload.put("task_type", model.getTaskType());

        JsonNode responseNode = mlEngineClient.predict(payload);

        PredictionResponseDto response = new PredictionResponseDto();
        response.setModelId(model.getId());
        response.setModelName(model.getModelName());

        if (responseNode.has("prediction")) {
            JsonNode pred = responseNode.get("prediction");
            if (pred.isNumber()) {
                response.setPrediction(pred.asDouble());
            } else {
                response.setPrediction(pred.asText());
            }
        }

        if (responseNode.has("confidence")) {
            response.setConfidence(responseNode.get("confidence").asDouble());
        }

        if (responseNode.has("probability") && responseNode.get("probability").isObject()) {
            Map<String, Double> probs = new HashMap<>();
            responseNode.get("probability").fields().forEachRemaining(entry ->
                    probs.put(entry.getKey(), entry.getValue().asDouble()));
            response.setProbability(probs);
        }

        return response;
    }
}
