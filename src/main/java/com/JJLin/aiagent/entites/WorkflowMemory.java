package com.JJLin.aiagent.entites;

import com.JJLin.aiagent.enums.VerificationFailureTag;
import lombok.Builder;
import lombok.Data;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

@Data
@Builder
public class WorkflowMemory {

    private String sessionId;

    @Builder.Default
    private List<String> resolvedFacts = new ArrayList<>();

    @Builder.Default
    private List<String> openQuestions = new ArrayList<>();

    private String latestPlanSummary;

    private String latestExecutionSummary;

    private VerificationFailureTag lastFailureTag;

    @Builder.Default
    private List<String> lastRequiredFixes = new ArrayList<>();

    private Instant updatedAt;
}