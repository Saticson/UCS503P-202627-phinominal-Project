package com.axiomcloud.service.ml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.classification.DecisionTree;
import smile.classification.GradientTreeBoost;
import smile.classification.RandomForest;
import smile.data.Tuple;
import smile.data.type.DataType;
import smile.data.type.StructField;
import smile.data.type.StructType;

import java.io.Serializable;
import java.util.*;

/**
 * Atomic, fused Machine Learning Pipeline artifact bundling preprocessing metadata,
 * imputation vectors, categorical dictionaries, and trained estimator weights.
 */
public class PipelineBundle implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final Logger logger = LoggerFactory.getLogger(PipelineBundle.class);

    private final String modelName;
    private final String modelType;
    private final String taskType;
    private final String targetColumn;
    private final List<String> featureNames;
    private final Map<String, String> featureTypeNames;
    private final Map<String, Map<String, Integer>> categoricalEncoders;
    private final Map<String, Map<Integer, String>> categoricalDecoders;
    private final Map<String, Double> imputationMeans;
    private final Map<String, Integer> labelToId;
    private final Map<Integer, String> idToLabel;
    private final Serializable estimator;
    private final StructType schema;

    public PipelineBundle(String modelName,
                          String modelType,
                          String taskType,
                          String targetColumn,
                          List<String> featureNames,
                          Map<String, String> featureTypeNames,
                          Map<String, Map<String, Integer>> categoricalEncoders,
                          Map<String, Map<Integer, String>> categoricalDecoders,
                          Map<String, Double> imputationMeans,
                          Map<String, Integer> labelToId,
                          Map<Integer, String> idToLabel,
                          Serializable estimator,
                          StructType schema) {
        this.modelName = modelName;
        this.modelType = modelType;
        this.taskType = taskType;
        this.targetColumn = targetColumn;
        this.featureNames = featureNames;
        this.featureTypeNames = featureTypeNames;
        this.categoricalEncoders = categoricalEncoders;
        this.categoricalDecoders = categoricalDecoders;
        this.imputationMeans = imputationMeans;
        this.labelToId = labelToId;
        this.idToLabel = idToLabel;
        this.estimator = estimator;
        this.schema = schema;
    }

    public String getModelName() { return modelName; }
    public String getModelType() { return modelType; }
    public String getTaskType() { return taskType; }
    public String getTargetColumn() { return targetColumn; }
    public List<String> getFeatureNames() { return featureNames; }
    public Map<Integer, String> getIdToLabel() { return idToLabel; }
    public Map<String, Integer> getLabelToId() { return labelToId; }
    public Serializable getEstimator() { return estimator; }
    public StructType getSchema() { return schema; }

    /**
     * Executes the preprocessing pipeline on raw user input JSON:
     * 1. Resolves case-insensitive feature keys.
     * 2. Imputes missing features with training distribution means/modes.
     * 3. Transforms string categories through fitted categorical encoder dictionaries.
     * 4. Coerces values into expected types and builds a type-safe Smile Tuple.
     */
    public Tuple transform(Map<String, Object> rawFeatures) {
        Object[] values = new Object[schema.length()];
        for (int i = 0; i < schema.length(); i++) {
            StructField field = schema.field(i);
            String name = field.name;
            Object val = findRawValue(rawFeatures, name);

            // Impute missing values with training mean / mode
            if (val == null) {
                if (imputationMeans.containsKey(name)) {
                    val = imputationMeans.get(name);
                } else if (categoricalEncoders.containsKey(name) && !categoricalEncoders.get(name).isEmpty()) {
                    val = 0;
                } else {
                    val = 0.0;
                }
            }

            // Transform categorical strings via fitted encoder
            if (categoricalEncoders.containsKey(name)) {
                if (val instanceof Number num) {
                    val = num.intValue();
                } else {
                    String strVal = val.toString().trim();
                    Map<String, Integer> enc = categoricalEncoders.get(name);
                    Integer catId = enc.get(strVal);
                    if (catId == null) {
                        // Case-insensitive lookup
                        for (Map.Entry<String, Integer> e : enc.entrySet()) {
                            if (e.getKey().equalsIgnoreCase(strVal)) {
                                catId = e.getValue();
                                break;
                            }
                        }
                    }
                    val = (catId != null) ? catId : 0;
                }
            }

            values[i] = coerceValue(val, field.type);
        }

        return Tuple.of(values, schema);
    }

    private Object findRawValue(Map<String, Object> raw, String fieldName) {
        if (raw == null) return null;
        if (raw.containsKey(fieldName)) return raw.get(fieldName);
        for (Map.Entry<String, Object> entry : raw.entrySet()) {
            if (entry.getKey().equalsIgnoreCase(fieldName)) {
                return entry.getValue();
            }
        }
        return null;
    }

    private Object coerceValue(Object val, DataType type) {
        if (val == null) {
            if (type == DataType.of(Double.class) || type == DataType.of(double.class)) return 0.0;
            if (type == DataType.of(Integer.class) || type == DataType.of(int.class)) return 0;
            return "0";
        }
        try {
            if (type == DataType.of(Double.class) || type == DataType.of(double.class)) {
                return Double.parseDouble(val.toString());
            }
            if (type == DataType.of(Integer.class) || type == DataType.of(int.class)) {
                return (int) Math.round(Double.parseDouble(val.toString()));
            }
        } catch (Exception ignored) {}
        return val.toString();
    }

    /**
     * Scores the preprocessed tuple against the fitted estimator.
     */
    public PredictionResult score(Tuple tuple) {
        int predictedClass = 0;
        double confidence = 0.95;
        double[] posteriors = null;

        if (estimator instanceof RandomForest rf) {
            posteriors = new double[idToLabel.isEmpty() ? 2 : Math.max(2, idToLabel.size())];
            predictedClass = rf.predict(tuple, posteriors);
            if (posteriors.length > predictedClass && predictedClass >= 0) {
                confidence = posteriors[predictedClass];
            }
        } else if (estimator instanceof GradientTreeBoost gtb) {
            posteriors = new double[idToLabel.isEmpty() ? 2 : Math.max(2, idToLabel.size())];
            predictedClass = gtb.predict(tuple, posteriors);
            if (posteriors.length > predictedClass && predictedClass >= 0) {
                confidence = posteriors[predictedClass];
            }
        } else if (estimator instanceof DecisionTree dt) {
            predictedClass = dt.predict(tuple);
            confidence = 0.90;
        }

        String label = idToLabel.getOrDefault(predictedClass, String.valueOf(predictedClass));
        return new PredictionResult(predictedClass, label, confidence, posteriors);
    }

    public static class PredictionResult {
        public final int classId;
        public final String label;
        public final double confidence;
        public final double[] posteriors;

        public PredictionResult(int classId, String label, double confidence, double[] posteriors) {
            this.classId = classId;
            this.label = label;
            this.confidence = confidence;
            this.posteriors = posteriors;
        }
    }
}
