package com.JJLin.aiagent.entites;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class ExecutionPlan {

    private String goal;

    private List<String> missingInfo;

    private Boolean blocked;

    private String blockReason;

    private List<ExecutionStep> steps;

    @Data
    public static class ExecutionStep {
        private String reason;
        private String actionType;
        private String targetService;
        private Map<String, Object> params;
        private String expectedOutput;
    }
}
