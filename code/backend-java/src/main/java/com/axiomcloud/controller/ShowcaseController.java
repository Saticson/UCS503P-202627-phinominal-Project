package com.axiomcloud.controller;

import com.axiomcloud.config.ShowcaseAuthenticationFilter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import org.springframework.context.annotation.Profile;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/showcase")
@Profile("showcase")
public class ShowcaseController {

    @PostMapping("/session")
    public ResponseEntity<Map<String, String>> createSession(HttpServletRequest request) {
        HttpSession session = request.getSession(true);
        session.setAttribute(ShowcaseAuthenticationFilter.SESSION_PRINCIPAL, "showcase-user");
        return ResponseEntity.ok(Map.of("message", "Showcase session started"));
    }
}
