package com.JJLin.aiagent.service;

import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ChatRequest;
import com.JJLin.aiagent.entites.ExecutionResult;
import com.JJLin.aiagent.entites.PlanDraft;
import com.JJLin.aiagent.entites.VerificationResult;
import com.JJLin.aiagent.entites.WorkflowMemory;
import com.JJLin.aiagent.entites.WorkflowState;
import com.JJLin.aiagent.enums.VerificationFailureTag;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import com.JJLin.aiagent.memory.WorkflowMemoryStore;
import org.springframework.ai.chat.messages.Message;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

@Service
public class WorkflowMemoryService {

    private final WorkflowMemoryStore workflowMemoryStore;

    public WorkflowMemoryService(WorkflowMemoryStore workflowMemoryStore) {
        this.workflowMemoryStore = workflowMemoryStore;
    }

    public WorkflowMemory getOrCreate(String sessionId) {
        return workflowMemoryStore.find(sessionId)
                .orElseGet(() -> workflowMemoryStore.save(WorkflowMemory.builder()
                        .sessionId(sessionId)
                        .resolvedFacts(new ArrayList<>())
                        .openQuestions(new ArrayList<>())
                        .lastRequiredFixes(new ArrayList<>())
                        .updatedAt(Instant.now())
                        .build()));
    }

    public WorkflowMemory updateAfterPlan(String sessionId, ChatRequest request, WorkflowState state, PlanDraft planDraft) {
        WorkflowMemory workflowMemory = getOrCreate(sessionId);
        workflowMemory.setOpenQuestions(merge(workflowMemory.getOpenQuestions(), safeList(planDraft.getMissingInfo())));
        workflowMemory.setResolvedFacts(merge(workflowMemory.getResolvedFacts(), safeList(planDraft.getAssumptions())));
        workflowMemory.setLatestPlanSummary(firstNonBlank(planDraft.getContextSummary(), summarizePlan(planDraft)));
        workflowMemory.setUpdatedAt(Instant.now());
        return workflowMemoryStore.save(workflowMemory);
    }

    public WorkflowMemory updateAfterExecutionStep(String sessionId, ActionSpec actionSpec, ExecutionResult executionResult) {
        WorkflowMemory workflowMemory = getOrCreate(sessionId);
        String summary = "%s -> %s".formatted(
                actionSpec.getActionType(),
                firstNonBlank(executionResult.getMessage(), String.valueOf(executionResult.getStatus())));
        workflowMemory.setLatestExecutionSummary(append(workflowMemory.getLatestExecutionSummary(), summary));
        workflowMemory.setUpdatedAt(Instant.now());
        return workflowMemoryStore.save(workflowMemory);
    }

    public WorkflowMemory updateAfterVerification(String sessionId, VerificationResult verificationResult) {
        WorkflowMemory workflowMemory = getOrCreate(sessionId);
        if (Boolean.TRUE.equals(verificationResult.getPass())) {
            workflowMemory.setLastFailureTag(VerificationFailureTag.NONE);
            workflowMemory.setLastRequiredFixes(new ArrayList<>());
        } else {
            workflowMemory.setLastFailureTag(verificationResult.getFailureTag());
            workflowMemory.setLastRequiredFixes(new ArrayList<>(safeList(verificationResult.getRequiredFixes())));
            workflowMemory.setOpenQuestions(merge(workflowMemory.getOpenQuestions(), safeList(verificationResult.getRequiredFixes())));
        }
        workflowMemory.setUpdatedAt(Instant.now());
        return workflowMemoryStore.save(workflowMemory);
    }

    public WorkflowMemory finalizeTurn(String sessionId, ChatRequest request, String reply, WorkflowState state, List<Message> recentMessages) {
        WorkflowMemory workflowMemory = getOrCreate(sessionId);
        if (state.getCurrentTaskStatus() == WorkflowTaskStatus.SUCCEEDED) {
            workflowMemory.setLastFailureTag(VerificationFailureTag.NONE);
            workflowMemory.setLastRequiredFixes(new ArrayList<>());
            workflowMemory.setOpenQuestions(new ArrayList<>());
        } else {
            workflowMemory.setLastFailureTag(state.getFailureReason());
            workflowMemory.setLastRequiredFixes(new ArrayList<>(safeList(state.getRequiredFixes())));
            workflowMemory.setOpenQuestions(merge(workflowMemory.getOpenQuestions(), safeList(state.getRequiredFixes())));
        }
        workflowMemory.setUpdatedAt(Instant.now());
        return workflowMemoryStore.save(workflowMemory);
    }

    private String summarizePlan(PlanDraft planDraft) {
        return safeList(planDraft.getSteps()).stream()
                .map(step -> firstNonBlank(step.getTitle(),
                        firstNonBlank(step.getDescription(), step.getExpectedOutput())))
                .reduce((left, right) -> left + " | " + right)
                .orElse(planDraft.getGoal());
    }

    private List<String> merge(List<String> left, List<String> right) {
        LinkedHashSet<String> merged = new LinkedHashSet<>();
        merged.addAll(safeList(left));
        merged.addAll(safeList(right));
        return new ArrayList<>(merged);
    }

    private String append(String existing, String next) {
        if (next == null || next.isBlank()) {
            return existing;
        }
        if (existing == null || existing.isBlank()) {
            return next;
        }
        return existing + " | " + next;
    }

    private String firstNonBlank(String first, String fallback) {
        return first != null && !first.isBlank() ? first : fallback;
    }

    private <T> List<T> safeList(List<T> values) {
        return values == null ? List.of() : values;
    }
}
