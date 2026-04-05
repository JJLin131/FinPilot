package com.JJLin.aiagent.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Data
@ConfigurationProperties(prefix = "ai.capability")
public class CapabilityProperties {

    private String baseUrl;

    private Duration connectTimeout = Duration.ofSeconds(5);

    private Duration readTimeout = Duration.ofSeconds(20);

    private Map<String, String> defaultHeaders = new LinkedHashMap<>();

    private String authHeaderName;

    private String authHeaderValue;

    private String workspaceRoot = ".";

    private Integer maxFileReadChars = 20000;

    private List<String> browserAllowedSchemes = new ArrayList<>(List.of("http", "https"));
}
