package com.JJLin.aiagent.entites;

import com.JJLin.aiagent.enums.WorkflowStage;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class ChatResponse {

    private WorkflowStage stage;

    private String reply;

    private PlanDraft planDraft;

    private ExecutionPlan executionPlan;

    private VerificationResult verificationResult;

    private List<ExecutionResult> executionResults;

    private WorkflowState workflowState;

    private String memorySummary;
}
