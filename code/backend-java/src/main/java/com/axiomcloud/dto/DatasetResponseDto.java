package com.axiomcloud.dto;

import com.axiomcloud.model.DatasetEntity;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

import java.time.OffsetDateTime;

public class DatasetResponseDto {

    private Integer id;
    private String name;
    private String filename;

    @JsonProperty("file_size")
    private Integer fileSize;

    @JsonProperty("num_rows")
    private Integer numRows;

    @JsonProperty("num_columns")
    private Integer numColumns;

    @JsonProperty("columns_info")
    private JsonNode columnsInfo;

    @JsonProperty("target_column")
    private String targetColumn;

    @JsonProperty("task_type")
    private String taskType;

    @JsonProperty("is_example")
    private Boolean isExample;

    @JsonProperty("preview_data")
    private JsonNode previewData;

    @JsonProperty("created_at")
    private OffsetDateTime createdAt;

    public DatasetResponseDto() {
    }

    public static DatasetResponseDto fromEntity(DatasetEntity entity) {
        DatasetResponseDto dto = new DatasetResponseDto();
        dto.setId(entity.getId());
        dto.setName(entity.getName());
        dto.setFilename(entity.getFilename());
        dto.setFileSize(entity.getFileSize());
        dto.setNumRows(entity.getNumRows());
        dto.setNumColumns(entity.getNumColumns());
        dto.setColumnsInfo(entity.getColumnsInfo());
        dto.setTargetColumn(entity.getTargetColumn());
        dto.setTaskType(entity.getTaskType());
        dto.setIsExample(entity.getIsExample());
        dto.setPreviewData(entity.getPreviewData());
        dto.setCreatedAt(entity.getCreatedAt());
        return dto;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public Integer getFileSize() {
        return fileSize;
    }

    public void setFileSize(Integer fileSize) {
        this.fileSize = fileSize;
    }

    public Integer getNumRows() {
        return numRows;
    }

    public void setNumRows(Integer numRows) {
        this.numRows = numRows;
    }

    public Integer getNumColumns() {
        return numColumns;
    }

    public void setNumColumns(Integer numColumns) {
        this.numColumns = numColumns;
    }

    public JsonNode getColumnsInfo() {
        return columnsInfo;
    }

    public void setColumnsInfo(JsonNode columnsInfo) {
        this.columnsInfo = columnsInfo;
    }

    public String getTargetColumn() {
        return targetColumn;
    }

    public void setTargetColumn(String targetColumn) {
        this.targetColumn = targetColumn;
    }

    public String getTaskType() {
        return taskType;
    }

    public void setTaskType(String taskType) {
        this.taskType = taskType;
    }

    public Boolean getIsExample() {
        return isExample;
    }

    public void setIsExample(Boolean isExample) {
        this.isExample = isExample;
    }

    public JsonNode getPreviewData() {
        return previewData;
    }

    public void setPreviewData(JsonNode previewData) {
        this.previewData = previewData;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(OffsetDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
