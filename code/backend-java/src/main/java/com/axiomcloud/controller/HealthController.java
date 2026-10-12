package com.axiomcloud.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class HealthController {

    @GetMapping("/api/health")
    public Map<String, String> health() {
        return Map.of("status", "ok");
    }

    @GetMapping("/")
    public Map<String, String> root() {
        return Map.of("message", "Axiom Cloud AI API - Java Spring Boot Backend");
    }
}
