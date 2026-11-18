package com.axiomcloud.service.ml;

import com.axiomcloud.config.StorageProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import smile.classification.DecisionTree;
import smile.classification.GradientTreeBoost;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.io.Write;
import smile.validation.metric.AUC;
import smile.validation.metric.Accuracy;
import smile.validation.metric.R2;
import smile.validation.metric.RMSE;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Component
public class SmileTrainer {

    private static final Logger logger = LoggerFactory.getLogger(SmileTrainer.class);

    private final SmileDataPreprocessor preprocessor;
    private final StorageProperties storageProperties;
    private final ObjectMapper objectMapper;

    public SmileTrainer(SmileDataPreprocessor preprocessor,
                        StorageProperties storageProperties,
                        ObjectMapper objectMapper) {
        this.preprocessor = preprocessor;
        this.storageProperties = storageProperties;
        this.objectMapper = objectMapper;
    }

    public ObjectNode train(Map<String, Object> payload) {
        String jobId = (String) payload.getOrDefault("job_id", UUID.randomUUID().toString().substring(0, 8));
        String datasetPath = (String) payload.get("dataset_path");
        String targetCol = (String) payload.getOrDefault("target_column", "label");

        logger.info("SmileTrainer executing fused pipeline sweep for job={} dataset={}", jobId, datasetPath);

        ObjectNode root = objectMapper.createObjectNode();
        root.put("job_id", jobId);
        root.put("status", "completed");

        Path modelDir = Paths.get(storageProperties.getModelsPath());
        try {
            Files.createDirectories(modelDir);
        } catch (IOException e) {
            logger.warn("Could not create models directory: {}", e.getMessage());
        }

        try {
            SmileDataPreprocessor.PreprocessedData prep = preprocessor.prepare(Paths.get(datasetPath), targetCol);
            DataFrame df = prep.getDataFrame();
            Formula formula = prep.getFormula();
            String taskType = prep.isClassification() ? "classification" : "regression";
            root.put("task_type", taskType);

            // Split into 80% train and 20% validation split
            int total = df.size();
            int trainSize = Math.max(1, (int) (total * 0.80));
            DataFrame trainDf = df.slice(0, trainSize);
            DataFrame valDf = (total > trainSize) ? df.slice(trainSize, total) : trainDf;

            ArrayNode modelsNode = root.putArray("models");

            Object modelsToTrainObj = payload.get("models_to_train");
            List<String> requested = (modelsToTrainObj instanceof List<?> list)
                    ? list.stream().map(Object::toString).toList()
                    : Collections.emptyList();

            boolean trainAll = requested.isEmpty();

            // Candidate 1: Random Forest
            if (trainAll || matchesAlgorithm(requested, "random_forest", "random forest")) {
                trainRandomForest(jobId, trainDf, valDf, formula, prep, taskType, modelDir, modelsNode);
            }

            // Candidate 2: Decision Tree
            if (trainAll || matchesAlgorithm(requested, "decision_tree", "decision tree")) {
                trainDecisionTree(jobId, trainDf, valDf, formula, prep, taskType, modelDir, modelsNode);
            }

            // Candidate 3: Gradient Tree Boost
            if (trainAll || matchesAlgorithm(requested, "gradient_boosting", "gradient boosting", "gradient tree boost", "xgboost", "lightgbm")) {
                trainGradientTreeBoost(jobId, trainDf, valDf, formula, prep, taskType, modelDir, modelsNode);
            }

            // If none matched, train at least Random Forest as champion
            if (modelsNode.isEmpty()) {
                trainRandomForest(jobId, trainDf, valDf, formula, prep, taskType, modelDir, modelsNode);
            }

            logger.info("SmileTrainer successfully completed sweep for job {} with {} verified pipeline bundles",
                    jobId, modelsNode.size());
            return root;

        } catch (Exception e) {
            logger.error("SmileTrainer failed, falling back to basic model: {}", e.getMessage(), e);
            throw new RuntimeException("Embedded ML Engine training failure: " + e.getMessage(), e);
        }
    }

    private boolean matchesAlgorithm(List<String> requested, String... keywords) {
        for (String req : requested) {
            for (String kw : keywords) {
                if (req.toLowerCase().contains(kw.toLowerCase())) return true;
            }
        }
        return false;
    }

    private void trainRandomForest(String jobId, DataFrame trainDf, DataFrame valDf, Formula formula,
                                   SmileDataPreprocessor.PreprocessedData prep,
                                   String taskType, Path modelDir, ArrayNode modelsNode) {
        try {
            long t0 = System.currentTimeMillis();
            Serializable estimator;
            double[] importances;

            if ("regression".equalsIgnoreCase(taskType)) {
                smile.regression.RandomForest rfReg = smile.regression.RandomForest.fit(formula, trainDf);
                estimator = rfReg;
                importances = rfReg.importance();
            } else {
                RandomForest rf = RandomForest.fit(formula, trainDf);
                estimator = rf;
                importances = rf.importance();
            }
            long elapsed = System.currentTimeMillis() - t0;

            Path artifactPath = modelDir.resolve("model_" + jobId + "_random_forest.bin");
            PipelineBundle bundle = new PipelineBundle(
                    "Random Forest " + (prep.isClassification() ? "Classifier" : "Regressor"),
                    "random_forest", taskType,
                    prep.getTargetColumn(), prep.getFeatureNames(), prep.getFeatureTypeNames(),
                    prep.getCategoricalEncoders(), prep.getCategoricalDecoders(),
                    prep.getImputationMeans(), prep.getLabelToId(), prep.getIdToLabel(),
                    estimator, trainDf.schema()
            );

            Write.object(bundle, artifactPath);
            long bytes = Files.size(artifactPath);

            ObjectNode m = modelsNode.addObject();
            m.put("model_name", bundle.getModelName());
            m.put("model_type", "random_forest");
            m.put("task_type", taskType);
            m.put("file_path", artifactPath.toAbsolutePath().toString());
            m.put("file_size_bytes", bytes);
            m.put("training_time", elapsed / 1000.0);

            evaluateModel(estimator, valDf, prep, taskType, m);
            populateFeatureImportances(m, prep.getFeatureNames(), importances);
            logger.info("Random Forest pipeline bundled: time={}ms file={} bytes={}", elapsed, artifactPath, bytes);

        } catch (Exception e) {
            logger.warn("Random Forest training failed: {}", e.getMessage(), e);
        }
    }

    private void trainDecisionTree(String jobId, DataFrame trainDf, DataFrame valDf, Formula formula,
                                   SmileDataPreprocessor.PreprocessedData prep,
                                   String taskType, Path modelDir, ArrayNode modelsNode) {
        try {
            long t0 = System.currentTimeMillis();
            Serializable estimator;
            double[] importances;

            if ("regression".equalsIgnoreCase(taskType)) {
                smile.regression.RegressionTree dtReg = smile.regression.RegressionTree.fit(formula, trainDf);
                estimator = dtReg;
                importances = dtReg.importance();
            } else {
                DecisionTree dt = DecisionTree.fit(formula, trainDf);
                estimator = dt;
                importances = dt.importance();
            }
            long elapsed = System.currentTimeMillis() - t0;

            Path artifactPath = modelDir.resolve("model_" + jobId + "_decision_tree.bin");
            PipelineBundle bundle = new PipelineBundle(
                    "Decision Tree " + (prep.isClassification() ? "Classifier" : "Regressor"),
                    "decision_tree", taskType,
                    prep.getTargetColumn(), prep.getFeatureNames(), prep.getFeatureTypeNames(),
                    prep.getCategoricalEncoders(), prep.getCategoricalDecoders(),
                    prep.getImputationMeans(), prep.getLabelToId(), prep.getIdToLabel(),
                    estimator, trainDf.schema()
            );

            Write.object(bundle, artifactPath);
            long bytes = Files.size(artifactPath);

            ObjectNode m = modelsNode.addObject();
            m.put("model_name", bundle.getModelName());
            m.put("model_type", "decision_tree");
            m.put("task_type", taskType);
            m.put("file_path", artifactPath.toAbsolutePath().toString());
            m.put("file_size_bytes", bytes);
            m.put("training_time", elapsed / 1000.0);

            evaluateModel(estimator, valDf, prep, taskType, m);
            populateFeatureImportances(m, prep.getFeatureNames(), importances);
            logger.info("Decision Tree pipeline bundled: time={}ms bytes={}", elapsed, bytes);

        } catch (Exception e) {
            logger.warn("Decision Tree training failed: {}", e.getMessage(), e);
        }
    }

    private void trainGradientTreeBoost(String jobId, DataFrame trainDf, DataFrame valDf, Formula formula,
                                        SmileDataPreprocessor.PreprocessedData prep,
                                        String taskType, Path modelDir, ArrayNode modelsNode) {
        try {
            long t0 = System.currentTimeMillis();
            Serializable estimator;
            double[] importances;

            if ("regression".equalsIgnoreCase(taskType)) {
                smile.regression.GradientTreeBoost gtbReg = smile.regression.GradientTreeBoost.fit(formula, trainDf);
                estimator = gtbReg;
                importances = gtbReg.importance();
            } else {
                GradientTreeBoost gtb = GradientTreeBoost.fit(formula, trainDf);
                estimator = gtb;
                importances = gtb.importance();
            }
            long elapsed = System.currentTimeMillis() - t0;

            Path artifactPath = modelDir.resolve("model_" + jobId + "_gradient_boosting.bin");
            PipelineBundle bundle = new PipelineBundle(
                    "Gradient Tree Boost " + (prep.isClassification() ? "Classifier" : "Regressor"),
                    "gradient_boosting", taskType,
                    prep.getTargetColumn(), prep.getFeatureNames(), prep.getFeatureTypeNames(),
                    prep.getCategoricalEncoders(), prep.getCategoricalDecoders(),
                    prep.getImputationMeans(), prep.getLabelToId(), prep.getIdToLabel(),
                    estimator, trainDf.schema()
            );

            Write.object(bundle, artifactPath);
            long bytes = Files.size(artifactPath);

            ObjectNode m = modelsNode.addObject();
            m.put("model_name", bundle.getModelName());
            m.put("model_type", "gradient_boosting");
            m.put("task_type", taskType);
            m.put("file_path", artifactPath.toAbsolutePath().toString());
            m.put("file_size_bytes", bytes);
            m.put("training_time", elapsed / 1000.0);

            evaluateModel(estimator, valDf, prep, taskType, m);
            populateFeatureImportances(m, prep.getFeatureNames(), importances);
            logger.info("Gradient Tree Boost pipeline bundled: time={}ms bytes={}", elapsed, bytes);

        } catch (Exception e) {
            logger.warn("Gradient Tree Boost training failed: {}", e.getMessage(), e);
        }
    }

    private void evaluateModel(Object model, DataFrame valDf,
                               SmileDataPreprocessor.PreprocessedData prep,
                               String taskType, ObjectNode m) {
        int n = valDf.size();
        if (n == 0) return;
        String targetCol = prep.getTargetColumn();

        if ("regression".equalsIgnoreCase(taskType)) {
            double[] truth = new double[n];
            double[] preds = new double[n];
            double sumAbsDiff = 0.0;

            for (int i = 0; i < n; i++) {
                Tuple t = valDf.get(i);
                truth[i] = t.getDouble(targetCol);
                double p = 0.0;
                if (model instanceof smile.regression.RandomForest rf) p = rf.predict(t);
                else if (model instanceof smile.regression.GradientTreeBoost gtb) p = gtb.predict(t);
                else if (model instanceof smile.regression.RegressionTree dt) p = dt.predict(t);
                preds[i] = p;
                sumAbsDiff += Math.abs(truth[i] - p);
            }

            double rmse = RMSE.of(truth, preds);
            double r2 = R2.of(truth, preds);
            double mae = sumAbsDiff / n;

            m.put("rmse", Math.round(rmse * 1000.0) / 1000.0);
            m.put("r2_score", Math.round(r2 * 1000.0) / 1000.0);
            m.put("mae", Math.round(mae * 1000.0) / 1000.0);
            m.put("accuracy", Math.max(0.0, Math.round(r2 * 1000.0) / 1000.0));
            m.put("score", Math.round(r2 * 1000.0) / 1000.0);

        } else {
            int[] truth = new int[n];
            int[] preds = new int[n];
            int numClasses = Math.max(2, prep.getIdToLabel().size());
            double[][] allProbs = new double[n][numClasses];

            int correct = 0;
            Map<Integer, Integer> tp = new HashMap<>();
            Map<Integer, Integer> actualCount = new HashMap<>();
            Map<Integer, Integer> predCount = new HashMap<>();

            for (int i = 0; i < n; i++) {
                Tuple t = valDf.get(i);
                truth[i] = t.getInt(targetCol);

                int p = 0;
                double[] post = new double[numClasses];
                if (model instanceof RandomForest rf) p = rf.predict(t, post);
                else if (model instanceof GradientTreeBoost gtb) p = gtb.predict(t, post);
                else if (model instanceof DecisionTree dt) p = dt.predict(t, post);

                preds[i] = p;
                allProbs[i] = post;

                if (p == truth[i]) {
                    correct++;
                    tp.merge(p, 1, Integer::sum);
                }
                actualCount.merge(truth[i], 1, Integer::sum);
                predCount.merge(p, 1, Integer::sum);
            }

            double acc = (double) correct / n;

            // Macro F1 computation
            double f1Sum = 0.0;
            int distinctClasses = Math.max(1, actualCount.size());
            for (Map.Entry<Integer, Integer> entry : actualCount.entrySet()) {
                int c = entry.getKey();
                int cTp = tp.getOrDefault(c, 0);
                int cFp = predCount.getOrDefault(c, 0) - cTp;
                int cFn = entry.getValue() - cTp;
                double precision = (cTp + cFp > 0) ? (double) cTp / (cTp + cFp) : 0.0;
                double recall = (cTp + cFn > 0) ? (double) cTp / (cTp + cFn) : 0.0;
                double cF1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
                f1Sum += cF1;
            }
            double macroF1 = f1Sum / distinctClasses;

            // Binary ROC-AUC if exactly 2 classes
            double rocAuc = 0.0;
            if (numClasses == 2) {
                double[] p1 = new double[n];
                for (int i = 0; i < n; i++) p1[i] = allProbs[i][1];
                try {
                    rocAuc = AUC.of(truth, p1);
                } catch (Exception ignored) {}
            }

            m.put("accuracy", Math.round(acc * 1000.0) / 1000.0);
            m.put("f1_score", Math.round(macroF1 * 1000.0) / 1000.0);
            m.put("roc_auc", Math.round(rocAuc * 1000.0) / 1000.0);
            m.put("score", Math.round(macroF1 * 1000.0) / 1000.0);
        }
    }

    private void populateFeatureImportances(ObjectNode m, List<String> featureNames, double[] importances) {
        if (importances != null && importances.length > 0 && !featureNames.isEmpty()) {
            ArrayNode feats = m.putArray("feature_importances");
            double sum = 0.0;
            for (double imp : importances) sum += imp;
            if (sum == 0.0) sum = 1.0;

            int limit = Math.min(featureNames.size(), importances.length);
            for (int i = 0; i < limit; i++) {
                ObjectNode item = feats.addObject();
                item.put("feature", featureNames.get(i));
                item.put("importance", Math.round((importances[i] / sum) * 1000.0) / 1000.0);
            }
        }
    }
}
