package com.axiomcloud.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;

@Configuration
public class FirebaseConfig {

    private static final Logger logger = LoggerFactory.getLogger(FirebaseConfig.class);

    @Value("${axiomcloud.firebase.credentials-json:}")
    private String credentialsJson;

    @Value("${axiomcloud.firebase.project-id:}")
    private String projectId;

    @PostConstruct
    public void initFirebase() {
        if (!FirebaseApp.getApps().isEmpty()) {
            return;
        }

        try {
            String rawJson = credentialsJson;
            if (rawJson == null || rawJson.isBlank()) {
                rawJson = System.getenv("FIREBASE_CREDENTIALS_JSON");
            }
            if (rawJson == null || rawJson.isBlank()) {
                rawJson = System.getenv("FIREBASE_CREDENTIALS");
            }

            if (rawJson != null && !rawJson.isBlank()) {
                // Strip quotes if wrapped
                rawJson = rawJson.trim();
                if ((rawJson.startsWith("'") && rawJson.endsWith("'")) ||
                    (rawJson.startsWith("\"") && rawJson.endsWith("\""))) {
                    rawJson = rawJson.substring(1, rawJson.length() - 1);
                }

                FirebaseOptions options = FirebaseOptions.builder()
                        .setCredentials(GoogleCredentials.fromStream(
                                new ByteArrayInputStream(rawJson.getBytes(StandardCharsets.UTF_8))))
                        .build();

                FirebaseApp.initializeApp(options);
                logger.info("Firebase Admin SDK initialized successfully via credentials JSON");
                return;
            }

            String envProjectId = (projectId != null && !projectId.isBlank()) ? projectId : System.getenv("FIREBASE_PROJECT_ID");
            if (envProjectId != null && !envProjectId.isBlank()) {
                FirebaseOptions options = FirebaseOptions.builder()
                        .setProjectId(envProjectId)
                        .build();
                FirebaseApp.initializeApp(options);
                logger.info("Firebase Admin SDK initialized with project ID: {}", envProjectId);
                return;
            }

            logger.warn("Firebase credentials not configured. Protected endpoints will reject requests.");
        } catch (Exception e) {
            logger.warn("Could not initialize Firebase Admin SDK: {}", e.getMessage());
        }
    }
}
