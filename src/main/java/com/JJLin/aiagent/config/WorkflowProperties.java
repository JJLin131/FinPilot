package com.JJLin.aiagent.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Data
@ConfigurationProperties(prefix = "ai.workflow")
public class WorkflowProperties {

    private int maxRetries = 3;
}
