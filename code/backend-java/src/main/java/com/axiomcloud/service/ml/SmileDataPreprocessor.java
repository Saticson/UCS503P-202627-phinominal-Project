package com.axiomcloud.service.ml;

import org.apache.commons.csv.CSVFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.type.DataType;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.data.vector.BaseVector;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.io.Read;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

@Component
public class SmileDataPreprocessor {

    private static final Logger logger = LoggerFactory.getLogger(SmileDataPreprocessor.class);

    public static class PreprocessedData {
        private final DataFrame dataFrame;
        private final Formula formula;
        private final String targetColumn;
        private final boolean classification;
        private final List<String> featureNames;
        private final Map<String, String> featureTypeNames;
        private final Map<String, Map<String, Integer>> categoricalEncoders;
        private final Map<String, Map<Integer, String>> categoricalDecoders;
        private final Map<String, Double> imputationMeans;
        private final Map<String, Integer> labelToId;
        private final Map<Integer, String> idToLabel;

        public PreprocessedData(DataFrame dataFrame, Formula formula, String targetColumn,
                                boolean classification, List<String> featureNames,
                                Map<String, String> featureTypeNames,
                                Map<String, Map<String, Integer>> categoricalEncoders,
                                Map<String, Map<Integer, String>> categoricalDecoders,
                                Map<String, Double> imputationMeans,
                                Map<String, Integer> labelToId,
                                Map<Integer, String> idToLabel) {
            this.dataFrame = dataFrame;
            this.formula = formula;
            this.targetColumn = targetColumn;
            this.classification = classification;
            this.featureNames = featureNames;
            this.featureTypeNames = featureTypeNames;
            this.categoricalEncoders = categoricalEncoders;
            this.categoricalDecoders = categoricalDecoders;
            this.imputationMeans = imputationMeans;
            this.labelToId = labelToId;
            this.idToLabel = idToLabel;
        }

        public DataFrame getDataFrame() { return dataFrame; }
        public Formula getFormula() { return formula; }
        public String getTargetColumn() { return targetColumn; }
        public boolean isClassification() { return classification; }
        public List<String> getFeatureNames() { return featureNames; }
        public Map<String, String> getFeatureTypeNames() { return featureTypeNames; }
        public Map<String, Map<String, Integer>> getCategoricalEncoders() { return categoricalEncoders; }
        public Map<String, Map<Integer, String>> getCategoricalDecoders() { return categoricalDecoders; }
        public Map<String, Double> getImputationMeans() { return imputationMeans; }
        public Map<String, Integer> getLabelToId() { return labelToId; }
        public Map<Integer, String> getIdToLabel() { return idToLabel; }
        public int numRows() { return dataFrame.size(); }
        public int numFeatures() { return featureNames.size(); }
    }

    public PreprocessedData prepare(Path csvPath, String targetColumn) throws IOException {
        logger.info("Loading and preprocessing tabular dataset: {}", csvPath);
        CSVFormat format = CSVFormat.DEFAULT.builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreHeaderCase(true)
                .setTrim(true)
                .setNullString("")
                .build();

        DataFrame raw = Read.csv(csvPath, format);
        if (raw == null || raw.isEmpty()) {
            throw new IllegalArgumentException("Dataset CSV is empty or cannot be parsed: " + csvPath);
        }

        // Validate target column exists in schema
        StructType schema = raw.schema();
        String actualTarget = null;
        for (StructField field : schema.fields()) {
            if (field.name.equalsIgnoreCase(targetColumn)) {
                actualTarget = field.name;
                break;
            }
        }

        if (actualTarget == null) {
            actualTarget = schema.field(schema.length() - 1).name;
            logger.warn("Target column '{}' not found, defaulting to last column: '{}'", targetColumn, actualTarget);
        }

        BaseVector targetVector = raw.column(actualTarget);
        boolean isClassification = true;
        Map<String, Integer> labelToId = new LinkedHashMap<>();
        Map<Integer, String> idToLabel = new LinkedHashMap<>();

        if (targetVector.type() == DataType.of(Double.class) || targetVector.type() == DataType.of(Float.class)) {
            Set<Double> uniqueVals = new HashSet<>();
            for (int i = 0; i < raw.size() && uniqueVals.size() < 15; i++) {
                if (targetVector.get(i) != null) {
                    uniqueVals.add(targetVector.getDouble(i));
                }
            }
            if (uniqueVals.size() > 10) {
                isClassification = false;
            }
        }

        Map<String, Map<String, Integer>> categoricalEncoders = new LinkedHashMap<>();
        Map<String, Map<Integer, String>> categoricalDecoders = new LinkedHashMap<>();
        Map<String, Double> imputationMeans = new LinkedHashMap<>();
        Map<String, String> featureTypeNames = new LinkedHashMap<>();
        List<String> featureNames = new ArrayList<>();

        BaseVector[] newCols = new BaseVector[schema.length()];

        for (int c = 0; c < schema.length(); c++) {
            StructField field = schema.field(c);
            String colName = field.name;
            BaseVector col = raw.column(c);

            if (colName.equals(actualTarget)) {
                if (isClassification) {
                    int[] encodedLabels = new int[raw.size()];
                    int nextId = 0;
                    for (int i = 0; i < raw.size(); i++) {
                        Object obj = col.get(i);
                        String val = (obj == null) ? "unknown" : obj.toString().trim();
                        if (!labelToId.containsKey(val)) {
                            labelToId.put(val, nextId);
                            idToLabel.put(nextId, val);
                            nextId++;
                        }
                        encodedLabels[i] = labelToId.get(val);
                    }
                    newCols[c] = IntVector.of(actualTarget, encodedLabels);
                } else {
                    double[] vals = new double[raw.size()];
                    for (int i = 0; i < raw.size(); i++) {
                        Object obj = col.get(i);
                        vals[i] = (obj instanceof Number n) ? n.doubleValue() : 0.0;
                    }
                    newCols[c] = DoubleVector.of(actualTarget, vals);
                }
            } else {
                featureNames.add(colName);
                featureTypeNames.put(colName, field.type.toString());

                if (col.type() == DataType.of(String.class)) {
                    Map<String, Integer> catEnc = new LinkedHashMap<>();
                    Map<Integer, String> catDec = new LinkedHashMap<>();
                    int[] encVals = new int[raw.size()];
                    int nextCat = 0;
                    for (int i = 0; i < raw.size(); i++) {
                        Object obj = col.get(i);
                        String val = (obj == null) ? "missing" : obj.toString().trim();
                        if (!catEnc.containsKey(val)) {
                            catEnc.put(val, nextCat);
                            catDec.put(nextCat, val);
                            nextCat++;
                        }
                        encVals[i] = catEnc.get(val);
                    }
                    categoricalEncoders.put(colName, catEnc);
                    categoricalDecoders.put(colName, catDec);
                    imputationMeans.put(colName, 0.0);
                    newCols[c] = IntVector.of(colName, encVals);
                } else {
                    double sum = 0.0;
                    int count = 0;
                    double[] numVals = new double[raw.size()];
                    for (int i = 0; i < raw.size(); i++) {
                        Object obj = col.get(i);
                        if (obj instanceof Number n) {
                            double d = n.doubleValue();
                            numVals[i] = d;
                            sum += d;
                            count++;
                        } else {
                            numVals[i] = 0.0;
                        }
                    }
                    double mean = (count > 0) ? (sum / count) : 0.0;
                    imputationMeans.put(colName, mean);
                    // Fill any missing with mean
                    for (int i = 0; i < raw.size(); i++) {
                        if (col.get(i) == null) {
                            numVals[i] = mean;
                        }
                    }
                    newCols[c] = DoubleVector.of(colName, numVals);
                }
            }
        }

        DataFrame processedDf = DataFrame.of(newCols);
        Formula formula = Formula.lhs(actualTarget);

        logger.info("Dataset preprocessed: {} rows, {} features (classification={})",
                processedDf.size(), featureNames.size(), isClassification);

        return new PreprocessedData(processedDf, formula, actualTarget, isClassification,
                featureNames, featureTypeNames, categoricalEncoders, categoricalDecoders,
                imputationMeans, labelToId, idToLabel);
    }
}
