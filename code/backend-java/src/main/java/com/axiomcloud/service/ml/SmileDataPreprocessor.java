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
import smile.data.vector.IntVector;
import smile.data.vector.StringVector;
import smile.data.vector.Vector;
import smile.io.Read;

import java.io.File;
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
        private final Map<String, Integer> labelToId;
        private final Map<Integer, String> idToLabel;

        public PreprocessedData(DataFrame dataFrame, Formula formula, String targetColumn,
                                boolean classification, List<String> featureNames,
                                Map<String, Integer> labelToId, Map<Integer, String> idToLabel) {
            this.dataFrame = dataFrame;
            this.formula = formula;
            this.targetColumn = targetColumn;
            this.classification = classification;
            this.featureNames = featureNames;
            this.labelToId = labelToId;
            this.idToLabel = idToLabel;
        }

        public DataFrame getDataFrame() { return dataFrame; }
        public Formula getFormula() { return formula; }
        public String getTargetColumn() { return targetColumn; }
        public boolean isClassification() { return classification; }
        public List<String> getFeatureNames() { return featureNames; }
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
            // Default to last column if specified target not found
            actualTarget = schema.field(schema.length() - 1).name;
            logger.warn("Target column '{}' not found, defaulting to last column: '{}'", targetColumn, actualTarget);
        }

        BaseVector targetVector = raw.column(actualTarget);
        boolean isClassification = true;
        Map<String, Integer> labelToId = new LinkedHashMap<>();
        Map<Integer, String> idToLabel = new LinkedHashMap<>();

        // Determine if target is continuous or categorical
        if (targetVector.type() == DataType.of(Double.class) || targetVector.type() == DataType.of(Float.class)) {
            // Check unique count to determine if continuous regression or binary
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

        DataFrame processedDf = raw;

        if (isClassification) {
            // Encode target column as discrete integers 0..K-1
            int[] encodedLabels = new int[raw.size()];
            int nextId = 0;
            for (int i = 0; i < raw.size(); i++) {
                Object obj = targetVector.get(i);
                String val = (obj == null) ? "unknown" : obj.toString().trim();
                if (!labelToId.containsKey(val)) {
                    labelToId.put(val, nextId);
                    idToLabel.put(nextId, val);
                    nextId++;
                }
                encodedLabels[i] = labelToId.get(val);
            }
            // Replace target column with IntVector
            BaseVector[] newCols = new BaseVector[schema.length()];
            for (int c = 0; c < schema.length(); c++) {
                if (schema.field(c).name.equals(actualTarget)) {
                    newCols[c] = IntVector.of(actualTarget, encodedLabels);
                } else {
                    newCols[c] = raw.column(c);
                }
            }
            processedDf = DataFrame.of(newCols);
        }

        // Collect feature names (all columns except target)
        List<String> featureNames = new ArrayList<>();
        for (StructField field : processedDf.schema().fields()) {
            if (!field.name.equals(actualTarget)) {
                featureNames.add(field.name);
            }
        }

        Formula formula = Formula.lhs(actualTarget);
        return new PreprocessedData(processedDf, formula, actualTarget, isClassification,
                featureNames, labelToId, idToLabel);
    }
}
