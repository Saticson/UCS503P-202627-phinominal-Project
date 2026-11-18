package com.axiomcloud.service.ml;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import smile.classification.DecisionTree;
import smile.classification.GradientTreeBoost;
import smile.classification.RandomForest;
import smile.data.Tuple;
import smile.data.type.DataType;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.io.Read;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class SmileInferenceEngine {

    private static final Logger logger = LoggerFactory.getLogger(SmileInferenceEngine.class);
    private final Map<String, Object> modelCache = new ConcurrentHashMap<>();
    private final ObjectMapper objectMapper;

    public SmileInferenceEngine(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public ObjectNode predict(String filePath, Map<String, Object> features, String taskType) {
        ObjectNode root = objectMapper.createObjectNode();

        if (filePath == null) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_REQUEST, "Model file path cannot be null");
        }

        Path pObj = Paths.get(filePath);
        if (!Files.exists(pObj)) {
            Path fname = pObj.getFileName();
            Path[] candidates = new Path[] {
                    Paths.get("storage/models", fname.toString()),
                    Paths.get("./storage/models", fname.toString()),
                    Paths.get("code/backend-java/storage/models", fname.toString()),
                    Paths.get("../storage/models", fname.toString()),
                    Paths.get("../../storage/models", fname.toString()),
                    Paths.get("code/storage/models", fname.toString())
            };
            for (Path c : candidates) {
                if (Files.exists(c)) {
                    pObj = c;
                    break;
                }
            }
        }

        if (!Files.exists(pObj)) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.NOT_FOUND,
                    "Model artifact file not found on disk: " + filePath);
        }

        final String resolvedPath = pObj.toAbsolutePath().toString();

        if (features == null || features.isEmpty()) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.BAD_REQUEST,
                    "Prediction features must be provided as a non-empty key-value map");
        }

        try {
            Object model = modelCache.computeIfAbsent(resolvedPath, p -> {
                try {
                    return Read.object(Paths.get(p));
                } catch (Exception e) {
                    logger.warn("Could not deserialize model at {}: {}", p, e.getMessage());
                    return null;
                }
            });

            if (model == null) {
                throw new org.springframework.web.server.ResponseStatusException(
                        org.springframework.http.HttpStatus.UNPROCESSABLE_ENTITY,
                        "Model artifact is corrupted or cannot be deserialized: " + filePath);
            }

            // ==========================================
            // MODERN FUSED PIPELINE EXECUTION
            // ==========================================
            if (model instanceof PipelineBundle bundle) {
                Tuple tuple = bundle.transform(features);
                PipelineBundle.PredictionResult res = bundle.score(tuple);

                root.put("prediction", res.label);
                root.put("confidence", Math.round(res.confidence * 1000.0) / 1000.0);

                ObjectNode probs = root.putObject("probability");
                if (res.posteriors != null && res.posteriors.length > 0) {
                    for (int k = 0; k < res.posteriors.length; k++) {
                        String classLabel = (bundle.getIdToLabel() != null)
                                ? bundle.getIdToLabel().getOrDefault(k, "class_" + k)
                                : "class_" + k;
                        probs.put(classLabel, Math.round(res.posteriors[k] * 1000.0) / 1000.0);
                    }
                } else {
                    probs.put("negative", Math.round((1.0 - res.confidence) * 1000.0) / 1000.0);
                    probs.put("positive", Math.round(res.confidence * 1000.0) / 1000.0);
                }

                logger.info("Executed fused PipelineBundle inference: model={} result={} conf={}",
                        bundle.getModelName(), res.label, res.confidence);
                return root;
            }

            // ==========================================
            // BACKWARDS COMPATIBLE LEGACY EXECUTION
            // ==========================================
            StructType schema = null;
            if (model instanceof RandomForest rf) schema = rf.schema();
            else if (model instanceof DecisionTree dt) schema = dt.schema();
            else if (model instanceof GradientTreeBoost gtb) schema = gtb.schema();

            if (schema == null) {
                throw new org.springframework.web.server.ResponseStatusException(
                        org.springframework.http.HttpStatus.UNPROCESSABLE_ENTITY,
                        "Unsupported model schema format: " + model.getClass().getName());
            }

            Object[] values = new Object[schema.length()];
            for (int i = 0; i < schema.length(); i++) {
                StructField field = schema.field(i);
                Object val = features != null ? features.get(field.name) : null;
                if (val == null && features != null) {
                    for (Map.Entry<String, Object> entry : features.entrySet()) {
                        if (entry.getKey().equalsIgnoreCase(field.name)) {
                            val = entry.getValue();
                            break;
                        }
                    }
                }
                values[i] = coerceValue(val, field.type);
            }

            Tuple tuple = Tuple.of(values, schema);
            int predictedClass = 0;
            double confidence = 0.95;

            if (model instanceof RandomForest rf) {
                predictedClass = rf.predict(tuple);
            } else if (model instanceof GradientTreeBoost gtb) {
                predictedClass = gtb.predict(tuple);
            } else if (model instanceof DecisionTree dt) {
                predictedClass = dt.predict(tuple);
            }

            root.put("prediction", predictedClass == 1 ? "positive" : (predictedClass == 0 ? "negative" : String.valueOf(predictedClass)));
            root.put("confidence", Math.round(confidence * 1000.0) / 1000.0);
            ObjectNode probs = root.putObject("probability");
            probs.put("negative", Math.round((1.0 - confidence) * 1000.0) / 1000.0);
            probs.put("positive", Math.round(confidence * 1000.0) / 1000.0);

            return root;

        } catch (org.springframework.web.server.ResponseStatusException rse) {
            throw rse;
        } catch (Exception e) {
            logger.warn("Live inference exception: {}", e.getMessage());
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.INTERNAL_SERVER_ERROR,
                    "Inference failed: " + e.getMessage(), e);
        }
    }

    private Object coerceValue(Object val, DataType type) {
        String typeStr = (type != null) ? type.toString().toLowerCase() : "";
        boolean isDouble = typeStr.contains("double") || typeStr.contains("float");
        boolean isInt = typeStr.contains("int") || typeStr.contains("byte") || typeStr.contains("short") || typeStr.contains("long");

        if (val == null) {
            if (isDouble) return 0.0;
            if (isInt) return 0;
            return "";
        }

        if (isDouble) {
            if (val instanceof Number n) return n.doubleValue();
            try {
                return Double.parseDouble(val.toString().trim());
            } catch (Exception e) {
                return 0.0;
            }
        }

        if (isInt) {
            if (val instanceof Number n) return n.intValue();
            try {
                return (int) Math.round(Double.parseDouble(val.toString().trim()));
            } catch (Exception e) {
                return 0;
            }
        }

        return val.toString();
    }
}
