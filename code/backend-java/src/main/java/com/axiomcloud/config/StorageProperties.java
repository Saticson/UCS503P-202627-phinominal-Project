package com.axiomcloud.config;

import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.File;

@Component
public class StorageProperties {

    @Value("${axiomcloud.storage.base-path:./storage}")
    private String basePath;

    @Value("${axiomcloud.storage.datasets-path:./storage/datasets}")
    private String datasetsPath;

    @Value("${axiomcloud.storage.models-path:./storage/models}")
    private String modelsPath;

    @Value("${axiomcloud.storage.reports-path:./storage/reports}")
    private String reportsPath;

    @PostConstruct
    public void init() {
        new File(basePath).mkdirs();
        new File(datasetsPath).mkdirs();
        new File(modelsPath).mkdirs();
        new File(reportsPath).mkdirs();
    }

    public String getBasePath() {
        return basePath;
    }

    public String getDatasetsPath() {
        return datasetsPath;
    }

    public String getModelsPath() {
        return modelsPath;
    }

    public String getReportsPath() {
        return reportsPath;
    }
}
