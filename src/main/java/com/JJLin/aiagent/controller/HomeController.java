package com.JJLin.aiagent.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
public class HomeController {

    @GetMapping("/")
    public Map<String, Object> home() {
        return Map.of(
                "service", "finance-ai-agent-service",
                "status", "UP",
                "message", "This project exposes REST APIs and does not include a web UI.",
                "endpoints", List.of(
                        "POST /api/finance/chat",
                        "POST /api/knowledge/bootstrap/resources",
                        "POST /api/knowledge/documents",
                        "POST /api/knowledge/evaluate",
                        "DELETE /api/knowledge/expired",
                        "GET /actuator/health"));
    }
}
