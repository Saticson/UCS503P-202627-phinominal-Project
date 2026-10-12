package com.axiomcloud.service;

import com.axiomcloud.config.StorageProperties;
import com.axiomcloud.dto.DatasetResponseDto;
import com.axiomcloud.model.DatasetEntity;
import com.axiomcloud.repository.DatasetRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.poi.ss.usermodel.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Service
public class DatasetService {

    private static final Logger logger = LoggerFactory.getLogger(DatasetService.class);
    private static final long MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB

    private final DatasetRepository datasetRepository;
    private final StorageProperties storageProperties;
    private final MlEngineClient mlEngineClient;
    private final ObjectMapper objectMapper;

    public DatasetService(DatasetRepository datasetRepository,
                          StorageProperties storageProperties,
                          MlEngineClient mlEngineClient,
                          ObjectMapper objectMapper) {
        this.datasetRepository = datasetRepository;
        this.storageProperties = storageProperties;
        this.mlEngineClient = mlEngineClient;
        this.objectMapper = objectMapper;
    }

    public List<DatasetResponseDto> listDatasets() {
        return datasetRepository.findAllByOrderByCreatedAtDesc()
                .stream()
                .map(DatasetResponseDto::fromEntity)
                .toList();
    }

    public DatasetEntity getDataset(int id) {
        return datasetRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Dataset not found"));
    }

    public DatasetResponseDto getDatasetDto(int id) {
        return DatasetResponseDto.fromEntity(getDataset(id));
    }

    @Transactional
    public DatasetResponseDto uploadDataset(MultipartFile file, String name, String targetColumn) {
        if (file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "File is empty");
        }

        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null || (!originalFilename.endsWith(".csv") && !originalFilename.endsWith(".xlsx") && !originalFilename.endsWith(".xls"))) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Only CSV and Excel files are supported");
        }

        if (file.getSize() > MAX_FILE_SIZE) {
            throw new ResponseStatusException(HttpStatus.PAYLOAD_TOO_LARGE, "File too large. Max 5MB allowed");
        }

        String fileId = UUID.randomUUID().toString().substring(0, 8);
        String safeName = fileId + "_" + originalFilename;
        Path targetPath = Paths.get(storageProperties.getDatasetsPath(), safeName);

        try {
            Files.copy(file.getInputStream(), targetPath);
        } catch (IOException e) {
            logger.error("Failed to store dataset file: {}", e.getMessage());
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to save file: " + e.getMessage());
        }

        // Parse dataset content for metadata
        try {
            DatasetMetadata metadata = analyzeDatasetFile(targetPath.toFile());

            DatasetEntity entity = new DatasetEntity();
            entity.setName(name != null && !name.isBlank() ? name : originalFilename);
            entity.setFilename(originalFilename);
            entity.setFilePath(targetPath.toAbsolutePath().toString());
            entity.setFileSize((int) file.getSize());
            entity.setNumRows(metadata.numRows);
            entity.setNumColumns(metadata.numColumns);
            entity.setColumnsInfo(metadata.columnsInfo);
            entity.setTargetColumn(targetColumn);
            entity.setIsExample(false);
            entity.setPreviewData(metadata.previewData);

            DatasetEntity saved = datasetRepository.save(entity);
            logger.info("Dataset uploaded successfully: id={} name={} rows={}", saved.getId(), saved.getName(), saved.getNumRows());
            return DatasetResponseDto.fromEntity(saved);
        } catch (Exception e) {
            try { Files.deleteIfExists(targetPath); } catch (Exception ignored) {}
            logger.error("Failed to parse dataset {}: {}", originalFilename, e.getMessage());
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY, "Could not parse dataset: " + e.getMessage());
        }
    }

    @Transactional
    public void deleteDataset(int id) {
        DatasetEntity entity = getDataset(id);
        if (entity.getFilePath() != null) {
            try {
                Files.deleteIfExists(Paths.get(entity.getFilePath()));
            } catch (Exception e) {
                logger.warn("Could not delete dataset file on disk: {}", e.getMessage());
            }
        }
        datasetRepository.delete(entity);
        logger.info("Deleted dataset id={}", id);
    }

    public JsonNode getQualityReport(int id) {
        DatasetEntity entity = getDataset(id);
        return mlEngineClient.qualityReport(entity.getFilePath());
    }

    public JsonNode getCleanPreview(int id, int previewRows) {
        DatasetEntity entity = getDataset(id);
        ObjectNode response = mlEngineClient.cleanPreview(entity.getFilePath(), previewRows).deepCopy();
        response.put("dataset_id", entity.getId());
        response.put("dataset_name", entity.getName());
        return response;
    }

    @Transactional
    public DatasetResponseDto cleanAndSave(int id, String nameSuffix) {
        DatasetEntity source = getDataset(id);
        String suffix = nameSuffix == null || nameSuffix.isBlank() ? "cleaned" : nameSuffix.replaceAll("[^A-Za-z0-9_-]", "_");
        String filename = source.getFilename();
        int dot = filename.lastIndexOf('.');
        String baseName = dot > 0 ? filename.substring(0, dot) : filename;
        String extension = dot > 0 ? filename.substring(dot).toLowerCase(Locale.ROOT) : ".csv";
        if (!extension.equals(".csv") && !extension.equals(".xlsx") && !extension.equals(".xls")) {
            extension = ".csv";
        }

        String cleanedFilename = baseName + "_" + suffix + extension;
        Path targetPath = Paths.get(storageProperties.getDatasetsPath(), UUID.randomUUID().toString().substring(0, 8) + "_" + cleanedFilename);
        mlEngineClient.cleanAndSave(source.getFilePath(), targetPath.toAbsolutePath().toString());
        if (!Files.isRegularFile(targetPath)) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "ML engine did not create cleaned dataset");
        }

        DatasetMetadata metadata = analyzeDatasetFile(targetPath.toFile());
        DatasetEntity cleaned = new DatasetEntity();
        cleaned.setName(source.getName() + " (" + suffix + ")");
        cleaned.setFilename(cleanedFilename);
        cleaned.setFilePath(targetPath.toAbsolutePath().toString());
        cleaned.setFileSize(Math.toIntExact(targetPath.toFile().length()));
        cleaned.setNumRows(metadata.numRows);
        cleaned.setNumColumns(metadata.numColumns);
        cleaned.setColumnsInfo(metadata.columnsInfo);
        cleaned.setTargetColumn(source.getTargetColumn());
        cleaned.setTaskType(source.getTaskType());
        cleaned.setIsExample(false);
        cleaned.setPreviewData(metadata.previewData);
        return DatasetResponseDto.fromEntity(datasetRepository.save(cleaned));
    }

    public JsonNode getLeakageReport(int id) {
        DatasetEntity entity = getDataset(id);
        ObjectNode response = mlEngineClient.leakageReport(entity.getId(), entity.getName(), entity.getFilePath(), entity.getTargetColumn()).deepCopy();
        response.put("dataset_id", entity.getId());
        response.put("dataset_name", entity.getName());
        response.put("target_column", entity.getTargetColumn());
        return response;
    }

    public JsonNode getDriftBaseline(int id) {
        DatasetEntity entity = getDataset(id);
        ObjectNode response = mlEngineClient.driftBaseline(entity.getFilePath(), entity.getTargetColumn()).deepCopy();
        response.put("dataset_id", entity.getId());
        response.put("dataset_name", entity.getName());
        response.put("target_column", entity.getTargetColumn());
        return response;
    }

    public JsonNode getProfile(int id) {
        DatasetEntity entity = getDataset(id);
        return mlEngineClient.datasetProfile(entity.getId(), entity.getName(), entity.getFilePath(), entity.getTargetColumn());
    }

    public JsonNode getEdaReport(int id) {
        DatasetEntity entity = getDataset(id);
        return mlEngineClient.edaReport(entity.getName(), entity.getFilePath(), entity.getTargetColumn());
    }

    public JsonNode getAnalyticsReport(int id, String targetColumn, String modelType, Integer modelId) {
        DatasetEntity entity = getDataset(id);
        String resolvedTarget = targetColumn != null ? targetColumn : entity.getTargetColumn();
        return mlEngineClient.analyticsReport(entity.getId(), entity.getName(), entity.getFilePath(), resolvedTarget, modelId, modelType);
    }

    @Transactional
    public DatasetResponseDto loadExampleDataset(String key) {
        // Built-in example datasets mapping
        Map<String, String> exampleNames = Map.of(
                "california_housing", "California Housing",
                "iris", "Iris Species",
                "titanic", "Titanic Survival",
                "diabetes", "Diabetes Progression"
        );

        String name = exampleNames.getOrDefault(key, key);

        Optional<DatasetEntity> existing = datasetRepository.findByIsExampleTrueAndName(name);
        if (existing.isPresent()) {
            return DatasetResponseDto.fromEntity(existing.get());
        }

        // Delegate loading of example dataset to ML engine or generate standard dataset
        String safeName = "example_" + key + ".csv";
        Path targetPath = Paths.get(storageProperties.getDatasetsPath(), safeName);

        // Generate built-in Iris dataset if file does not exist
        if (!Files.exists(targetPath)) {
            generateBuiltinExample(key, targetPath);
        }

        DatasetMetadata metadata = analyzeDatasetFile(targetPath.toFile());

        DatasetEntity entity = new DatasetEntity();
        entity.setName(name);
        entity.setFilename(safeName);
        entity.setFilePath(targetPath.toAbsolutePath().toString());
        entity.setFileSize((int) targetPath.toFile().length());
        entity.setNumRows(metadata.numRows);
        entity.setNumColumns(metadata.numColumns);
        entity.setColumnsInfo(metadata.columnsInfo);
        entity.setTargetColumn(getDefaultTarget(key));
        entity.setTaskType(getDefaultTaskType(key));
        entity.setIsExample(true);
        entity.setPreviewData(metadata.previewData);

        DatasetEntity saved = datasetRepository.save(entity);
        return DatasetResponseDto.fromEntity(saved);
    }

    private String getDefaultTarget(String key) {
        return switch (key) {
            case "california_housing" -> "MedHouseVal";
            case "iris" -> "species";
            case "titanic" -> "survived";
            case "diabetes" -> "target";
            default -> "target";
        };
    }

    private String getDefaultTaskType(String key) {
        return switch (key) {
            case "california_housing", "diabetes" -> "regression";
            default -> "classification";
        };
    }

    private void generateBuiltinExample(String key, Path path) {
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path))) {
            if ("iris".equalsIgnoreCase(key)) {
                writer.println("sepal_length,sepal_width,petal_length,petal_width,species");
                writer.println("5.1,3.5,1.4,0.2,setosa");
                writer.println("4.9,3.0,1.4,0.2,setosa");
                writer.println("4.7,3.2,1.3,0.2,setosa");
                writer.println("7.0,3.2,4.7,1.4,versicolor");
                writer.println("6.4,3.2,4.5,1.5,versicolor");
                writer.println("6.9,3.1,4.9,1.5,versicolor");
                writer.println("6.3,3.3,6.0,2.5,virginica");
                writer.println("5.8,2.7,5.1,1.9,virginica");
                writer.println("7.1,3.0,5.9,2.1,virginica");
                writer.println("6.3,2.9,5.6,1.8,virginica");
            } else {
                writer.println("feature1,feature2,feature3,target");
                for (int i = 0; i < 50; i++) {
                    writer.printf(Locale.US, "%.2f,%.2f,%.2f,%d%n", Math.random() * 10, Math.random() * 5, Math.random() * 20, (int)(Math.random() * 2));
                }
            }
        } catch (IOException e) {
            logger.error("Failed to generate example dataset: {}", e.getMessage());
        }
    }

    private DatasetMetadata analyzeDatasetFile(File file) {
        if (file.getName().endsWith(".csv")) {
            return analyzeCsv(file);
        } else {
            return analyzeExcel(file);
        }
    }

    private DatasetMetadata analyzeCsv(File file) {
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                rows.add(parseCsvLine(line));
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV: " + e.getMessage(), e);
        }

        if (rows.isEmpty()) {
            throw new IllegalArgumentException("CSV file is empty");
        }

        String[] headers = rows.get(0);
        int numColumns = headers.length;
        int numRows = rows.size() - 1;

        ArrayNode columnsInfo = objectMapper.createArrayNode();
        ArrayNode previewData = objectMapper.createArrayNode();

        // Build preview data (first 10 rows)
        for (int i = 1; i < Math.min(11, rows.size()); i++) {
            String[] row = rows.get(i);
            ObjectNode rowObj = objectMapper.createObjectNode();
            for (int col = 0; col < headers.length; col++) {
                String val = col < row.length ? row[col] : "";
                rowObj.put(headers[col], val);
            }
            previewData.add(rowObj);
        }

        // Analyze columns
        for (int col = 0; col < headers.length; col++) {
            String colName = headers[col];
            int nullCount = 0;
            Set<String> uniqueVals = new HashSet<>();
            List<Double> numericVals = new ArrayList<>();
            ArrayNode samples = objectMapper.createArrayNode();

            for (int r = 1; r < rows.size(); r++) {
                String[] row = rows.get(r);
                String val = col < row.length ? row[col].trim() : "";
                if (val.isEmpty() || "null".equalsIgnoreCase(val) || "nan".equalsIgnoreCase(val)) {
                    nullCount++;
                } else {
                    uniqueVals.add(val);
                    if (samples.size() < 5) samples.add(val);
                    try {
                        numericVals.add(Double.parseDouble(val));
                    } catch (NumberFormatException ignored) {}
                }
            }

            boolean isNumeric = numericVals.size() > (numRows - nullCount) * 0.8 && !numericVals.isEmpty();

            ObjectNode colInfo = objectMapper.createObjectNode();
            colInfo.put("name", colName);
            colInfo.put("dtype", isNumeric ? "float64" : "object");
            colInfo.put("null_count", nullCount);
            colInfo.put("unique_count", uniqueVals.size());
            colInfo.set("sample_values", samples);

            if (isNumeric && !numericVals.isEmpty()) {
                double min = numericVals.stream().mapToDouble(v -> v).min().orElse(0);
                double max = numericVals.stream().mapToDouble(v -> v).max().orElse(0);
                double mean = numericVals.stream().mapToDouble(v -> v).average().orElse(0);

                ObjectNode stats = objectMapper.createObjectNode();
                stats.put("min", min);
                stats.put("max", max);
                stats.put("mean", mean);
                colInfo.set("stats", stats);
            }

            columnsInfo.add(colInfo);
        }

        DatasetMetadata meta = new DatasetMetadata();
        meta.numRows = numRows;
        meta.numColumns = numColumns;
        meta.columnsInfo = columnsInfo;
        meta.previewData = previewData;
        return meta;
    }

    private DatasetMetadata analyzeExcel(File file) {
        try (Workbook workbook = WorkbookFactory.create(file)) {
            Sheet sheet = workbook.getSheetAt(0);
            Row headerRow = sheet.getRow(0);
            if (headerRow == null) throw new IllegalArgumentException("Excel sheet is empty");

            int numColumns = headerRow.getLastCellNum();
            int numRows = sheet.getLastRowNum();

            ArrayNode columnsInfo = objectMapper.createArrayNode();
            ArrayNode previewData = objectMapper.createArrayNode();

            // Preview first 10 rows
            for (int r = 1; r <= Math.min(10, numRows); r++) {
                Row row = sheet.getRow(r);
                if (row == null) continue;
                ObjectNode rowObj = objectMapper.createObjectNode();
                for (int c = 0; c < numColumns; c++) {
                    Cell cell = row.getCell(c);
                    String val = cell != null ? cell.toString() : "";
                    String header = headerRow.getCell(c) != null ? headerRow.getCell(c).toString() : "col_" + c;
                    rowObj.put(header, val);
                }
                previewData.add(rowObj);
            }

            for (int c = 0; c < numColumns; c++) {
                String header = headerRow.getCell(c) != null ? headerRow.getCell(c).toString() : "col_" + c;
                ObjectNode colInfo = objectMapper.createObjectNode();
                colInfo.put("name", header);
                colInfo.put("dtype", "object");
                colInfo.put("null_count", 0);
                colInfo.put("unique_count", 0);
                colInfo.set("sample_values", objectMapper.createArrayNode());
                columnsInfo.add(colInfo);
            }

            DatasetMetadata meta = new DatasetMetadata();
            meta.numRows = numRows;
            meta.numColumns = numColumns;
            meta.columnsInfo = columnsInfo;
            meta.previewData = previewData;
            return meta;
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse Excel file: " + e.getMessage(), e);
        }
    }

    private String[] parseCsvLine(String line) {
        List<String> tokens = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '\"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                tokens.add(sb.toString().trim());
                sb.setLength(0);
            } else {
                sb.append(c);
            }
        }
        tokens.add(sb.toString().trim());
        return tokens.toArray(new String[0]);
    }

    private static class DatasetMetadata {
        int numRows;
        int numColumns;
        JsonNode columnsInfo;
        JsonNode previewData;
    }
}
