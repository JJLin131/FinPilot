package com.JJLin.aiagent.entites;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class PlanDraft {

    private String goal;

    private String planType;

    private List<String> assumptions;

    private List<String> missingInfo;

    private List<PlanStep> steps;

    private List<String> stopConditions;

    private Boolean directResponse;

    private Boolean needsExecution;

    private String directResponseText;

    private String contextSummary;

    @Data
    public static class PlanStep {
        private String title;
        private String description;
        private List<String> prerequisites;
        private List<String> successCriteria;
        private String actionType;
        private String targetService;
        private Map<String, Object> params;
        private String expectedOutput;
    }
}
