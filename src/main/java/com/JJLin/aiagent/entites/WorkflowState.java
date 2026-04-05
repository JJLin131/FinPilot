package com.JJLin.aiagent.entites;

import com.JJLin.aiagent.enums.VerificationFailureTag;
import com.JJLin.aiagent.enums.WorkflowStage;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class WorkflowState {

    private String userId;

    private String chatId;

    private String taskId;

    private String parentTaskId;

    private String query;

    private WorkflowStage stage;

    private List<WorkflowStage> stageHistory = new ArrayList<>();

    private WorkflowTaskStatus currentTaskStatus;

    private Integer retries;

    private Integer maxRetries;

    private String contextSummary;

    private PlanDraft planDraft;

    private ExecutionPlan executionPlan;

    private VerificationResult lastVerificationResult;

    private VerificationFailureTag failureReason;

    private String failureMessage;

    private List<String> requiredFixes = new ArrayList<>();

    private List<ExecutionResult> executionResults = new ArrayList<>();
}
