package com.JJLin.aiagent.service;

import com.JJLin.aiagent.capability.CapabilityDefinition;
import com.JJLin.aiagent.capability.CapabilityParameter;
import com.JJLin.aiagent.config.WorkflowProperties;
import com.JJLin.aiagent.entites.WorkflowMemory;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ChatRequest;
import com.JJLin.aiagent.entites.ChatResponse;
import com.JJLin.aiagent.entites.ExecutionPlan;
import com.JJLin.aiagent.entites.ExecutionResult;
import com.JJLin.aiagent.entites.PlanDraft;
import com.JJLin.aiagent.entites.UserProfile;
import com.JJLin.aiagent.enums.VerificationFailureTag;
import com.JJLin.aiagent.entites.VerificationResult;
import com.JJLin.aiagent.enums.VerificationType;
import com.JJLin.aiagent.enums.WorkflowStage;
import com.JJLin.aiagent.entites.WorkflowState;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class WorkflowOrchestratorService {

    private static final Logger log = LoggerFactory.getLogger(WorkflowOrchestratorService.class);
    private static final int RETRIEVAL_LIMIT = 5;

    private final ChatClient orchestratorAgent;
    private final ChatClient planAgent;
    private final ChatClient executorAgent;
    private final ChatClient verifyAgent;
    private final CapabilityDispatcher capabilityDispatcher;
    private final WorkflowProperties workflowProperties;
    private final ChatMemory conversationMemory;
    private final WorkflowMemoryService workflowMemoryService;
    private final MemoryContextAssembler memoryContextAssembler;
    private final UserProfileService userProfileService;
    private final UserLongTermMemoryService userLongTermMemoryService;
    private final SystemOperationMemoryService systemOperationMemoryService;

    public WorkflowOrchestratorService(
            @Qualifier("orchestratorAgent") ChatClient orchestratorAgent,
            @Qualifier("planAgent") ChatClient planAgent,
            @Qualifier("executorAgent") ChatClient executorAgent,
            @Qualifier("verifyAgent") ChatClient verifyAgent,
            CapabilityDispatcher capabilityDispatcher,
            WorkflowProperties workflowProperties,
            ChatMemory conversationMemory,
            WorkflowMemoryService workflowMemoryService,
            MemoryContextAssembler memoryContextAssembler,
            UserProfileService userProfileService,
            UserLongTermMemoryService userLongTermMemoryService,
            SystemOperationMemoryService systemOperationMemoryService) {
        this.orchestratorAgent = orchestratorAgent;
        this.planAgent = planAgent;
        this.executorAgent = executorAgent;
        this.verifyAgent = verifyAgent;
        this.capabilityDispatcher = capabilityDispatcher;
        this.workflowProperties = workflowProperties;
        this.conversationMemory = conversationMemory;
        this.workflowMemoryService = workflowMemoryService;
        this.memoryContextAssembler = memoryContextAssembler;
        this.userProfileService = userProfileService;
        this.userLongTermMemoryService = userLongTermMemoryService;
        this.systemOperationMemoryService = systemOperationMemoryService;
    }

    public ChatResponse chat(ChatRequest request) {
        WorkflowState state = initializeState(request);
        String sessionId = request.getChatId();
        WorkflowMemory workflowMemory = workflowMemoryService.getOrCreate(sessionId);
        UserProfile userProfile = userProfileService.getByUserId(request.getUserId());
        List<String> userLongTermMemories = userLongTermMemoryService.findRelevant(
                request.getUserId(), request.getContent(), RETRIEVAL_LIMIT);
        List<String> systemOperationMemories = systemOperationMemoryService.findRelevant(
                request.getUserId(), inferOperationType(request.getContent(), null), RETRIEVAL_LIMIT);
        List<Message> recentMessages = conversationMemory.get(sessionId);

        userProfileService.updateFromRequestAsync(request, userProfile);
        userLongTermMemoryService.extractFromRequestAsync(request, userProfile);

        if (isDirectRequest(request.getContent())) {
            transitionTo(state, WorkflowStage.RESPOND);
            String reply = orchestratorAgent.prompt()
                    .user(memoryContextAssembler.assembleForResponse(
                            request.getContent(),
                            workflowMemory,
                            userProfile,
                            userLongTermMemories,
                            recentMessages,
                            null,
                            "Reply briefly to this simple user message."))
                    .call()
                    .content();
            state.setCurrentTaskStatus(WorkflowTaskStatus.SUCCEEDED);
            return finalizeTurn(sessionId, request, state, reply);
        }

        PlanDraft planDraft = null;
        VerificationResult planVerification = null;
        String retryFeedback = null;
        int maxAttempts = Math.max(1, workflowProperties.getMaxRetries() + 1);

        for (int attempt = 0; attempt < maxAttempts; attempt++) {
            state.setRetries(attempt);
            transitionTo(state, WorkflowStage.EXPLORE);
            planDraft = buildPlanDraft(
                    request,
                    retryFeedback,
                    workflowMemory,
                    userProfile,
                    userLongTermMemories,
                    systemOperationMemories,
                    recentMessages);
            state.setPlanDraft(planDraft);

            if (planDraft == null) {
                return fail(state, request, workflowMemory, VerificationFailureTag.VALIDATION_ERROR,
                        "Plan agent returned no structured plan.", Collections.emptyList(), VerificationType.PLAN);
            }

            transitionTo(state, WorkflowStage.PLAN);
            state.setContextSummary(planDraft.getContextSummary());
            workflowMemory = workflowMemoryService.updateAfterPlan(sessionId, request, state, planDraft);

            if (Boolean.TRUE.equals(planDraft.getDirectResponse())
                    && !Boolean.TRUE.equals(defaultTrue(planDraft.getNeedsExecution()))) {
                transitionTo(state, WorkflowStage.RESPOND);
                state.setCurrentTaskStatus(WorkflowTaskStatus.SUCCEEDED);
                String reply = firstNonBlank(planDraft.getDirectResponseText(),
                        composeDirectReply(
                                request.getContent(),
                                planDraft,
                                workflowMemory,
                                userProfile,
                                userLongTermMemories,
                                systemOperationMemories,
                                recentMessages));
                return finalizeTurn(sessionId, request, state, reply);
            }


            transitionTo(state, WorkflowStage.PLAN_VERIFY);
            planVerification = normalizePlanVerification(
                    planDraft,
                    verifyPlan(
                            request,
                            planDraft,
                            workflowMemory,
                            userProfile,
                            userLongTermMemories,
                            systemOperationMemories,
                            recentMessages));
            state.setLastVerificationResult(planVerification);
            state.setRequiredFixes(safeList(planVerification.getRequiredFixes()));
            workflowMemory = workflowMemoryService.updateAfterVerification(sessionId, planVerification);

            if (Boolean.TRUE.equals(planVerification.getPass())) {
                break;
            }

            if (attempt + 1 >= maxAttempts || !canRetryPlan(planVerification)) {
                return fail(state, request, workflowMemory,
                        defaultTag(planVerification.getFailureTag()),
                        firstNonBlank(planVerification.getSummary(), "Plan verification failed."),
                        safeList(planVerification.getRequiredFixes()),
                        VerificationType.PLAN);
            }

            retryFeedback = buildRetryFeedback(planVerification);
        }

        if (planDraft == null) {
            return fail(state, request, workflowMemory, VerificationFailureTag.VALIDATION_ERROR,
                    "Workflow finished without a plan draft.", Collections.emptyList(), VerificationType.PLAN);
        }

        transitionTo(state, WorkflowStage.EXECUTE);
        ExecutionPlan executionPlan = buildExecutionPlan(
                request,
                planDraft,
                workflowMemory,
                userProfile,
                userLongTermMemories,
                recentMessages);
        executionPlan = bindExecutionPlanParams(
                request,
                planDraft,
                executionPlan,
                workflowMemory,
                userProfile,
                userLongTermMemories,
                recentMessages);
        executionPlan = normalizeExecutionPlan(executionPlan);
        state.setExecutionPlan(executionPlan);

        if (executionPlan == null) {
            return fail(state, request, workflowMemory, VerificationFailureTag.VALIDATION_ERROR,
                    "Executor agent returned no structured execution plan.", Collections.emptyList(), VerificationType.EXECUTION);
        }

        List<String> executionMissingInfo = normalizeMissingInfo(executionPlan.getMissingInfo());
        if (Boolean.TRUE.equals(executionPlan.getBlocked()) && !executionMissingInfo.isEmpty()) {
            return fail(state, request, workflowMemory, VerificationFailureTag.MISSING_DATA,
                    buildMissingInfoMessage(executionMissingInfo), executionMissingInfo, VerificationType.EXECUTION);
        }

        List<String> executionPlanIssues = validateExecutionPlan(executionPlan);
        if (!executionPlanIssues.isEmpty()) {
            VerificationFailureTag failureTag = determineExecutionPlanFailureTag(executionPlanIssues);
            return fail(state, request, workflowMemory, failureTag,
                    "Executor output is not executable. Required fixes: " + String.join("; ", executionPlanIssues),
                    executionPlanIssues,
                    VerificationType.EXECUTION);
        }

        List<ExecutionResult> executionResults = new ArrayList<>();
        int stepIndex = 0;
        for (ExecutionPlan.ExecutionStep step : safeList(executionPlan.getSteps())) {
            ActionSpec actionSpec = new ActionSpec();
            actionSpec.setTaskId(state.getTaskId() + "-step-" + (++stepIndex));
            actionSpec.setActionType(step.getActionType());
            actionSpec.setTargetService(step.getTargetService());
            actionSpec.setParams(step.getParams());
            actionSpec.setExpectedOutput(step.getExpectedOutput());
            ExecutionResult executionResult = capabilityDispatcher.dispatch(actionSpec);
            executionResults.add(executionResult);
            workflowMemory = workflowMemoryService.updateAfterExecutionStep(sessionId, actionSpec, executionResult);
        }
        state.setExecutionResults(executionResults);

        transitionTo(state, WorkflowStage.EXECUTION_VERIFY);
        VerificationResult verificationResult = normalizeExecutionVerification(
                executionResults,
                verifyExecution(
                        request,
                        planDraft,
                        executionResults,
                        workflowMemory,
                        userProfile,
                        userLongTermMemories,
                        systemOperationMemories,
                        recentMessages));
        state.setLastVerificationResult(verificationResult);
        state.setRequiredFixes(safeList(verificationResult.getRequiredFixes()));
        workflowMemory = workflowMemoryService.updateAfterVerification(sessionId, verificationResult);

        if (Boolean.TRUE.equals(verificationResult.getPass())) {
            transitionTo(state, WorkflowStage.RESPOND);
            state.setCurrentTaskStatus(WorkflowTaskStatus.SUCCEEDED);
            String reply = orchestratorAgent.prompt()
                    .user(memoryContextAssembler.assembleForResponse(
                            request.getContent(),
                            workflowMemory,
                            userProfile,
                            userLongTermMemories,
                            recentMessages,
                            "Execution results: " + executionResults,
                            "Use the verified execution results to answer the user."))
                    .call()
                    .content();
            return finalizeTurn(sessionId, request, state, reply);
        }

        return fail(state, request, workflowMemory,
                defaultTag(verificationResult.getFailureTag()),
                buildExecutionFailureMessage(verificationResult, executionResults),
                safeList(verificationResult.getRequiredFixes()),
                VerificationType.EXECUTION);
    }

    private WorkflowState initializeState(ChatRequest request) {
        WorkflowState state = new WorkflowState();
        state.setUserId(request.getUserId());
        state.setChatId(request.getChatId());
        state.setQuery(request.getContent());
        state.setTaskId("wf-" + UUID.randomUUID());
        state.setParentTaskId(request.getChatId());
        state.setRetries(0);
        state.setMaxRetries(workflowProperties.getMaxRetries());
        state.setCurrentTaskStatus(WorkflowTaskStatus.RUNNING);
        transitionTo(state, WorkflowStage.INTAKE);
        return state;
    }

    private PlanDraft buildPlanDraft(
            ChatRequest request,
            String retryFeedback,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<String> systemOperationMemories,
            List<Message> recentMessages) {
        PlanDraft planDraft = planAgent.prompt()
                .user(memoryContextAssembler.assembleForPlan(
                        request.getContent(),
                        workflowMemory,
                        userProfile,
                        userLongTermMemories,
                        recentMessages,
                        "Build a PlanDraft for this request. Keep the steps intent-level and implementation-agnostic. "
                                + "Describe what should happen, any business prerequisites, success criteria, and the expected outputs. "
                                + "Do not choose handlers, target services, action types, or concrete parameter names in the plan. "
                                + "If information is missing, populate missingInfo in business language and set needsExecution carefully. "
                                + "If no downstream action is needed, set directResponse=true. Retry feedback: " + firstNonBlank(retryFeedback, "none")))
                .call()
                .entity(PlanDraft.class);
        return sanitizePlanDraft(planDraft);
    }

    private ExecutionPlan buildExecutionPlan(
            ChatRequest request,
            PlanDraft planDraft,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<Message> recentMessages) {
        return executorAgent.prompt()
                .user(memoryContextAssembler.assembleForExecution(
                        request.getContent(),
                        workflowMemory,
                        userProfile,
                        userLongTermMemories,
                        recentMessages,
                         "Plan: " + planDraft,
                         "Convert this intent-level plan into an execution plan. Supported capabilities: "
                                 + capabilityDispatcher.describeCapabilities()
                                 + ". Choose only supported actionType values, use only documented params, "
                                 + "and return blocked=true only when a required capability parameter for the chosen action is still missing. "
                                 + "Always include the intended execution step even when blocked so the system can validate the chosen handler. "
                                 + "Do not ask for optional preferences just because they could improve quality; they are non-blocking unless the chosen capability marks them required."))
                .call()
                .entity(ExecutionPlan.class);
    }

    private ExecutionPlan bindExecutionPlanParams(
            ChatRequest request,
            PlanDraft planDraft,
            ExecutionPlan executionPlan,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<Message> recentMessages) {
        if (executionPlan == null) {
            return null;
        }
        if (safeList(executionPlan.getSteps()).isEmpty()) {
            return executionPlan;
        }
        ExecutionPlan boundPlan = executorAgent.prompt()
                .user(memoryContextAssembler.assembleForExecution(
                        request.getContent(),
                        workflowMemory,
                        userProfile,
                        userLongTermMemories,
                        recentMessages,
                        "Plan: " + planDraft + ". Draft execution plan: " + executionPlan,
                        "Bind user semantics to execution params before validation. "
                                + "For each chosen execution step, infer and fill both required and optional params when the values are already available from the user request, plan context, user profile, long-term memory, or recent conversation. "
                                + "Keep the chosen action if it still fits. "
                                + "If the user explicitly stated a usable product category phrase such as running shoes or comedy movies, bind that phrase directly to category rather than leaving it empty. "
                                + "Only leave blocked=true if a required parameter is genuinely unavailable after this binding step. "
                                + "Do not ask for optional preferences or restate non-required missing information."))
                .call()
                .entity(ExecutionPlan.class);
        return applyDeterministicParamBindings(request, planDraft, boundPlan == null ? executionPlan : boundPlan);
    }

    private VerificationResult verifyPlan(
            ChatRequest request,
            PlanDraft planDraft,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<String> systemOperationMemories,
            List<Message> recentMessages) {
        WorkflowMemory memoryForPlanVerification = planVerificationMemoryView(workflowMemory);
        return verifyAgent.prompt()
                .user(memoryContextAssembler.assembleForVerification(
                        request.getContent(),
                        memoryForPlanVerification,
                        userProfile,
                        systemOperationMemories,
                        "Plan: " + planDraft,
                        "verificationType=" + VerificationType.PLAN
                                + ". Validate whether this plan is complete, coherent, and forms a reasonable business-level workflow. "
                                + "Do not check handler availability, actionType support, concrete parameter completeness, or execution-time missing preferences in plan verification. "
                                + "Do not fail a plan only because a similar tool failed in previous turns. "
                                + "Historical tool failures are execution-risk hints, not plan-invalidating evidence. "
                                + "Fail the plan only for current-plan defects such as contradictory steps, missing essential business steps, or a broken workflow that cannot close logically."))
                .call()
                .entity(VerificationResult.class);
    }

    private VerificationResult verifyExecution(
            ChatRequest request,
            PlanDraft planDraft,
            List<ExecutionResult> executionResults,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<String> systemOperationMemories,
            List<Message> recentMessages) {
        return verifyAgent.prompt()
                .user(memoryContextAssembler.assembleForVerification(
                        request.getContent(),
                        workflowMemory,
                        userProfile,
                        systemOperationMemories,
                        "Plan: " + planDraft + ". Execution results: " + executionResults,
                        "verificationType=" + VerificationType.EXECUTION
                                + ". Validate whether these execution results are enough for a final answer."))
                .call()
                .entity(VerificationResult.class);
    }

    private VerificationResult normalizePlanVerification(PlanDraft planDraft, VerificationResult verificationResult) {
        VerificationResult normalized = ensureVerification(verificationResult, VerificationType.PLAN);
        if (safeList(planDraft.getSteps()).isEmpty() && Boolean.TRUE.equals(defaultTrue(planDraft.getNeedsExecution()))) {
            return failedVerification(VerificationType.PLAN, VerificationFailureTag.PLAN_GAP,
                    "Plan requires execution but returned no intent-level steps.",
                    List.of("Add at least one business step or set directResponse=true."));
        }


        return normalized;
    }

    private ExecutionPlan normalizeExecutionPlan(ExecutionPlan executionPlan) {
        if (executionPlan == null) {
            return null;
        }

        List<ExecutionPlan.ExecutionStep> steps = safeList(executionPlan.getSteps());
        if (steps.isEmpty()) {
            executionPlan.setMissingInfo(normalizeMissingInfo(executionPlan.getMissingInfo()));
            return executionPlan;
        }

        LinkedHashSet<String> requiredMissingInfo = new LinkedHashSet<>();
        boolean inspectedDefinitions = false;

        for (ExecutionPlan.ExecutionStep step : steps) {
            if (step == null || step.getActionType() == null || step.getActionType().isBlank()) {
                continue;
            }
            CapabilityDefinition definition = resolveCapabilityDefinition(step.getActionType());
            if (definition == null) {
                continue;
            }
            inspectedDefinitions = true;
            for (CapabilityParameter parameter : safeList(definition.getParameters())) {
                if (!parameter.isRequired() || hasAnyParameterValue(step.getParams(), parameter)) {
                    continue;
                }
                requiredMissingInfo.add(requiredParamLabel(parameter));
            }
        }

        if (!requiredMissingInfo.isEmpty()) {
            executionPlan.setBlocked(Boolean.TRUE);
            executionPlan.setMissingInfo(new ArrayList<>(requiredMissingInfo));
            if (executionPlan.getBlockReason() == null || executionPlan.getBlockReason().isBlank()) {
                executionPlan.setBlockReason("Missing required capability parameters for the selected action.");
            }
            return executionPlan;
        }

        if (inspectedDefinitions) {
            executionPlan.setBlocked(Boolean.FALSE);
            executionPlan.setBlockReason(null);
            executionPlan.setMissingInfo(Collections.emptyList());
            return executionPlan;
        }

        executionPlan.setMissingInfo(normalizeMissingInfo(executionPlan.getMissingInfo()));
        return executionPlan;
    }

    private PlanDraft sanitizePlanDraft(PlanDraft planDraft) {
        if (planDraft == null) {
            return null;
        }
        for (PlanDraft.PlanStep step : safeList(planDraft.getSteps())) {
            step.setActionType(null);
            step.setTargetService(null);
            step.setParams(null);
        }
        return planDraft;
    }

    private VerificationResult normalizeExecutionVerification(
            List<ExecutionResult> executionResults, VerificationResult verificationResult) {
        VerificationResult normalized = ensureVerification(verificationResult, VerificationType.EXECUTION);

        List<ExecutionResult> failedResults = safeList(executionResults).stream()
                .filter(result -> result.getStatus() == WorkflowTaskStatus.FAILED)
                .toList();
        if (!failedResults.isEmpty()) {
            VerificationFailureTag failureTag = failedResults.stream()
                    .map(this::failureTagForExecution)
                    .filter(tag -> tag != VerificationFailureTag.NONE)
                    .findFirst()
                    .orElse(VerificationFailureTag.TOOL_FAILURE);
            return failedVerification(VerificationType.EXECUTION, failureTag,
                    "Execution produced failed steps.",
                    failedResults.stream()
                            .map(result -> "%s failed with %s".formatted(
                                    result.getActionType(), firstNonBlank(result.getErrorCode(), result.getMessage())))
                            .toList());
        }

        boolean noPayload = safeList(executionResults).stream().allMatch(result -> result.getPayload() == null);
        if (noPayload) {
            return failedVerification(VerificationType.EXECUTION, VerificationFailureTag.TOOL_FAILURE,
                    "Execution finished without payloads.", List.of("Check downstream capability responses."));
        }

        return normalized;
    }

    private VerificationResult ensureVerification(VerificationResult verificationResult, VerificationType verificationType) {
        if (verificationResult == null) {
            return failedVerification(verificationType, VerificationFailureTag.VALIDATION_ERROR,
                    "Verify agent returned null.", List.of("Re-run verification for " + verificationType + "."));
        }
        verificationResult.setVerificationType(verificationType);
        if (verificationResult.getFailureTag() == null) {
            verificationResult.setFailureTag(Boolean.TRUE.equals(verificationResult.getPass())
                    ? VerificationFailureTag.NONE : VerificationFailureTag.VALIDATION_ERROR);
        }
        if (verificationResult.getRequiredFixes() == null) {
            verificationResult.setRequiredFixes(new ArrayList<>());
        }
        if (verificationResult.getFindings() == null) {
            verificationResult.setFindings(new ArrayList<>());
        }
        return verificationResult;
    }

    private VerificationResult failedVerification(
            VerificationType verificationType,
            VerificationFailureTag failureTag,
            String summary,
            List<String> requiredFixes) {
        VerificationResult verificationResult = new VerificationResult();
        verificationResult.setPass(Boolean.FALSE);
        verificationResult.setVerificationType(verificationType);
        verificationResult.setFailureTag(failureTag);
        verificationResult.setSummary(summary);
        verificationResult.setRequiredFixes(new ArrayList<>(safeList(requiredFixes)));
        verificationResult.setFindings(new ArrayList<>());
        return verificationResult;
    }

    private boolean canRetryPlan(VerificationResult verificationResult) {
        return verificationResult.getFailureTag() == VerificationFailureTag.PLAN_GAP
                || verificationResult.getFailureTag() == VerificationFailureTag.VALIDATION_ERROR;
    }

    private String buildRetryFeedback(VerificationResult verificationResult) {
        return "Previous plan verification failed.\n"
                + "Summary: " + firstNonBlank(verificationResult.getSummary(), "unknown") + "\n"
                + "Required fixes: " + String.join("; ", safeList(verificationResult.getRequiredFixes()));
    }

    private String buildMissingInfoMessage(List<String> missingInfo) {
        return "I need a bit more information before I can execute this workflow: "
                + String.join(", ", safeList(missingInfo)) + ".";
    }

    private List<String> validateExecutionPlan(ExecutionPlan executionPlan) {
        if (Boolean.TRUE.equals(executionPlan.getBlocked())) {
            return Collections.emptyList();
        }
        if (safeList(executionPlan.getSteps()).isEmpty()) {
            return List.of("Executor produced no executable steps.");
        }

        List<String> issues = new ArrayList<>();
        for (ExecutionPlan.ExecutionStep step : safeList(executionPlan.getSteps())) {
            if (step.getActionType() == null || step.getActionType().isBlank()) {
                issues.add("Execution step is missing actionType.");
                continue;
            }
            if (!capabilityDispatcher.supports(step.getActionType())) {
                issues.add("Unsupported actionType: " + step.getActionType());
                continue;
            }
            ActionSpec actionSpec = new ActionSpec();
            actionSpec.setActionType(step.getActionType());
            actionSpec.setTargetService(step.getTargetService());
            actionSpec.setParams(step.getParams());
            issues.addAll(safeList(capabilityDispatcher.validateAction(actionSpec)));
        }
        return issues;
    }

    private VerificationFailureTag determineExecutionPlanFailureTag(List<String> issues) {
        if (issues.stream().anyMatch(issue -> issue != null && issue.startsWith("Unsupported actionType"))) {
            return VerificationFailureTag.UNSUPPORTED_ACTION;
        }
        if (issues.stream().anyMatch(issue -> issue != null && issue.contains("Missing required param"))) {
            return VerificationFailureTag.MISSING_DATA;
        }
        return VerificationFailureTag.VALIDATION_ERROR;
    }

    private List<String> normalizeMissingInfo(List<String> missingInfo) {
        LinkedHashSet<String> normalized = new LinkedHashSet<>();
        for (String item : safeList(missingInfo)) {
            if (item == null) {
                continue;
            }
            String trimmed = item.trim();
            if (!trimmed.isBlank()) {
                normalized.add(trimmed);
            }
        }
        return new ArrayList<>(normalized);
    }

    private CapabilityDefinition resolveCapabilityDefinition(String actionType) {
        if (actionType == null || actionType.isBlank()) {
            return null;
        }
        var definition = capabilityDispatcher.definition(actionType);
        return definition == null ? null : definition.orElse(null);
    }

    private boolean hasAnyParameterValue(Map<String, Object> params, CapabilityParameter parameter) {
        Map<String, Object> safeParams = params == null ? Collections.emptyMap() : params;
        for (String name : parameterNames(parameter)) {
            Object value = safeParams.get(name);
            if (value == null) {
                continue;
            }
            if (value instanceof String stringValue && stringValue.isBlank()) {
                continue;
            }
            return true;
        }
        return false;
    }

    private String requiredParamLabel(CapabilityParameter parameter) {
        return "Required parameter '" + parameter.getName() + "'";
    }

    private List<String> parameterNames(CapabilityParameter parameter) {
        List<String> names = new ArrayList<>();
        names.add(parameter.getName());
        if (parameter.getAliases() != null) {
            names.addAll(parameter.getAliases());
        }
        return names;
    }

    private ExecutionPlan applyDeterministicParamBindings(
            ChatRequest request,
            PlanDraft planDraft,
            ExecutionPlan executionPlan) {
        if (executionPlan == null) {
            return null;
        }
        String categoryHint = inferProductCategory(request, planDraft);
        String keywordHint = inferKeyword(request, planDraft);
        for (ExecutionPlan.ExecutionStep step : safeList(executionPlan.getSteps())) {
            if (step == null || step.getActionType() == null || step.getActionType().isBlank()) {
                continue;
            }
            Map<String, Object> params = step.getParams() == null ? new java.util.LinkedHashMap<>() : new java.util.LinkedHashMap<>(step.getParams());
            CapabilityDefinition definition = resolveCapabilityDefinition(step.getActionType());
            if (definition == null) {
                step.setParams(params);
                continue;
            }
            for (CapabilityParameter parameter : safeList(definition.getParameters())) {
                if (hasAnyParameterValue(params, parameter)) {
                    continue;
                }
                if (isCategoryParameter(parameter) && categoryHint != null) {
                    params.put(parameter.getName(), categoryHint);
                    continue;
                }
                if (isKeywordParameter(parameter) && keywordHint != null) {
                    params.put(parameter.getName(), keywordHint);
                    continue;
                }
                if (isUserIdParameter(parameter) && request.getUserId() != null && !request.getUserId().isBlank()) {
                    params.put(parameter.getName(), request.getUserId());
                }
            }
            step.setParams(params);
        }
        return executionPlan;
    }

    private String inferProductCategory(ChatRequest request, PlanDraft planDraft) {
        String combined = combinedContext(request, planDraft).toLowerCase();
        if (containsAny(combined, "\u8dd1\u6b65\u978b", "running shoe", "running shoes")) {
            return "running shoes";
        }
        if (containsAny(combined, "\u7bee\u7403\u978b", "basketball shoe", "basketball shoes")) {
            return "basketball shoes";
        }
        if (containsAny(combined, "\u4f11\u95f2\u978b", "casual shoe", "casual shoes")) {
            return "casual shoes";
        }
        if (containsAny(combined, "\u76ae\u978b", "formal shoe", "formal shoes", "dress shoes")) {
            return "formal shoes";
        }
        if (containsAny(combined, "\u978b", "shoe", "shoes", "sneaker", "sneakers")) {
            return "shoes";
        }
        if (containsAny(combined, "\u559c\u5267\u7535\u5f71", "comedy movie", "comedy movies")) {
            return "comedy movies";
        }
        if (containsAny(combined, "\u52a8\u4f5c\u7535\u5f71", "action movie", "action movies")) {
            return "action movies";
        }
        if (containsAny(combined, "\u7535\u5f71", "movie", "movies", "film", "films")) {
            return "movies";
        }
        if (containsAny(combined, "\u624b\u673a", "phone", "phones", "smartphone", "smartphones")) {
            return "phones";
        }
        return null;
    }

    private String inferKeyword(ChatRequest request, PlanDraft planDraft) {
        String categoryHint = inferProductCategory(request, planDraft);
        return categoryHint == null ? null : categoryHint;
    }

    private boolean isCategoryParameter(CapabilityParameter parameter) {
        return parameterNames(parameter).stream().anyMatch(name -> {
            String normalized = name.toLowerCase();
            return normalized.equals("category") || normalized.equals("itemcategory") || normalized.equals("itemcatagoty");
        });
    }

    private boolean isKeywordParameter(CapabilityParameter parameter) {
        return parameterNames(parameter).stream().anyMatch(name -> {
            String normalized = name.toLowerCase();
            return normalized.equals("keyword") || normalized.equals("query") || normalized.equals("q");
        });
    }

    private boolean isUserIdParameter(CapabilityParameter parameter) {
        return parameterNames(parameter).stream().anyMatch(name -> {
            String normalized = name.toLowerCase();
            return normalized.equals("userid") || normalized.equals("accountid");
        });
    }

    private String combinedContext(ChatRequest request, PlanDraft planDraft) {
        return firstNonBlank(request.getContent(), "") + " "
                + (planDraft == null ? "" : firstNonBlank(planDraft.getGoal(), "")) + " "
                + (planDraft == null ? "" : firstNonBlank(planDraft.getContextSummary(), ""));
    }

    private boolean containsAny(String source, String... candidates) {
        if (source == null) {
            return false;
        }
        String normalizedSource = source.toLowerCase();
        for (String candidate : candidates) {
            if (normalizedSource.contains(candidate.toLowerCase())) {
                return true;
            }
        }
        return false;
    }

    private String buildExecutionFailureMessage(VerificationResult verificationResult, List<ExecutionResult> executionResults) {
        if (!safeList(verificationResult.getRequiredFixes()).isEmpty()) {
            return firstNonBlank(verificationResult.getSummary(), "Execution verification failed.")
                    + " Required fixes: " + String.join("; ", verificationResult.getRequiredFixes());
        }
        String failedDetails = safeList(executionResults).stream()
                .filter(result -> result.getStatus() == WorkflowTaskStatus.FAILED)
                .map(result -> "%s(%s)".formatted(result.getActionType(),
                        firstNonBlank(result.getErrorCode(), result.getMessage())))
                .collect(Collectors.joining(", "));
        return firstNonBlank(verificationResult.getSummary(), "Execution verification failed.")
                + (failedDetails.isBlank() ? "" : " Failed steps: " + failedDetails);
    }

    private String composeDirectReply(
            String userRequest,
            PlanDraft planDraft,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<String> systemOperationMemories,
            List<Message> recentMessages) {
        return orchestratorAgent.prompt()
                .user(memoryContextAssembler.assembleForResponse(
                        userRequest,
                        workflowMemory,
                        userProfile,
                        userLongTermMemories,
                        recentMessages,
                        "Plan context: " + firstNonBlank(planDraft.getContextSummary(), planDraft.getGoal()),
                        "Respond directly to the user without executing downstream tools."))
                .call()
                .content();
    }

    private ChatResponse fail(
            WorkflowState state,
            ChatRequest request,
            WorkflowMemory workflowMemory,
            VerificationFailureTag failureTag,
            String message,
            List<String> requiredFixes,
            VerificationType verificationType) {
        transitionTo(state, WorkflowStage.FAILSAFE);
        state.setCurrentTaskStatus(WorkflowTaskStatus.FAILED);
        state.setFailureReason(failureTag);
        state.setFailureMessage(message);
        state.setRequiredFixes(new ArrayList<>(safeList(requiredFixes)));
        state.setLastVerificationResult(failedVerification(verificationType, failureTag, message, requiredFixes));
        workflowMemory = workflowMemoryService.updateAfterVerification(request.getChatId(), state.getLastVerificationResult());
        systemOperationMemoryService.record(
                request.getUserId(),
                inferOperationType(request.getContent(), state.getPlanDraft()),
                message);
        return finalizeTurn(request.getChatId(), request, state, message);
    }

    private ChatResponse finalizeTurn(String sessionId, ChatRequest request, WorkflowState state, String reply) {
        conversationMemory.add(sessionId, List.of(new UserMessage(request.getContent()), new AssistantMessage(reply)));
        WorkflowMemory workflowMemory = workflowMemoryService.finalizeTurn(
                sessionId,
                request,
                reply,
                state,
                conversationMemory.get(sessionId));
        return buildResponse(state, reply, workflowMemory);
    }

    private ChatResponse buildResponse(WorkflowState state, String reply, WorkflowMemory workflowMemory) {
        return ChatResponse.builder()
                .stage(state.getStage())
                .reply(reply)
                .planDraft(state.getPlanDraft())
                .executionPlan(state.getExecutionPlan())
                .verificationResult(state.getLastVerificationResult())
                .executionResults(state.getExecutionResults())
                .workflowState(state)
                .memorySummary(summarizeWorkflowMemory(workflowMemory))
                .build();
    }

    private String summarizeWorkflowMemory(WorkflowMemory workflowMemory) {
        if (workflowMemory == null) {
            return null;
        }
        List<String> parts = new ArrayList<>();
        if (workflowMemory.getLastFailureTag() != null && workflowMemory.getLastFailureTag() != VerificationFailureTag.NONE) {
            parts.add("lastFailureTag=" + workflowMemory.getLastFailureTag().name());
        }
        if (!safeList(workflowMemory.getLastRequiredFixes()).isEmpty()) {
            parts.add("requiredFixes=" + String.join("; ", workflowMemory.getLastRequiredFixes()));
        }
        if (workflowMemory.getLatestPlanSummary() != null && !workflowMemory.getLatestPlanSummary().isBlank()) {
            parts.add("latestPlan=" + workflowMemory.getLatestPlanSummary());
        }
        if (workflowMemory.getLatestExecutionSummary() != null && !workflowMemory.getLatestExecutionSummary().isBlank()) {
            parts.add("latestExecution=" + workflowMemory.getLatestExecutionSummary());
        }
        if (!safeList(workflowMemory.getResolvedFacts()).isEmpty()) {
            parts.add("resolvedFacts=" + String.join("; ", workflowMemory.getResolvedFacts()));
        }
        if (!safeList(workflowMemory.getOpenQuestions()).isEmpty()) {
            parts.add("openQuestions=" + String.join("; ", workflowMemory.getOpenQuestions()));
        }
        return parts.isEmpty() ? null : String.join(" | ", parts);
    }

    private WorkflowMemory planVerificationMemoryView(WorkflowMemory workflowMemory) {
        if (workflowMemory == null) {
            return null;
        }
        return WorkflowMemory.builder()
                .sessionId(workflowMemory.getSessionId())
                .resolvedFacts(new ArrayList<>(safeList(workflowMemory.getResolvedFacts())))
                .openQuestions(new ArrayList<>(safeList(workflowMemory.getOpenQuestions())))
                .latestPlanSummary(workflowMemory.getLatestPlanSummary())
                .latestExecutionSummary(null)
                .lastFailureTag(null)
                .lastRequiredFixes(new ArrayList<>())
                .updatedAt(workflowMemory.getUpdatedAt())
                .build();
    }

    private void transitionTo(WorkflowState state, WorkflowStage stage) {
        state.setStage(stage);
        state.getStageHistory().add(stage);
        log.info("workflow task={} chatId={} stage={}", state.getTaskId(), state.getChatId(), stage);
    }

    private VerificationFailureTag failureTagForExecution(ExecutionResult result) {
        if ("UNSUPPORTED_ACTION".equals(result.getErrorCode())) {
            return VerificationFailureTag.UNSUPPORTED_ACTION;
        }
        if ("MISSING_PARAM".equals(result.getErrorCode())) {
            return VerificationFailureTag.MISSING_DATA;
        }
        return VerificationFailureTag.TOOL_FAILURE;
    }

    private String inferOperationType(String content, PlanDraft planDraft) {
        if (planDraft != null && !safeList(planDraft.getSteps()).isEmpty()) {
            String actionType = safeList(planDraft.getSteps()).get(0).getActionType();
            return actionType == null || actionType.isBlank() ? "GENERAL" : actionType;
        }
        String normalized = content == null ? "" : content.toLowerCase();
        if (normalized.contains("order")) {
            return "CHECK_ORDER";
        }
        if (normalized.contains("browser")) {
            return "OPEN_BROWSER_URL";
        }
        if (normalized.contains("file")) {
            return "READ_FILE";
        }
        return "SEARCH_PRODUCTS";
    }

    private <T> List<T> safeList(List<T> values) {
        return values == null ? Collections.emptyList() : values;
    }

    private String firstNonBlank(String first, String fallback) {
        return first != null && !first.isBlank() ? first : fallback;
    }

    private Boolean defaultTrue(Boolean value) {
        return value != null ? value : Boolean.TRUE;
    }

    private VerificationFailureTag defaultTag(VerificationFailureTag failureTag) {
        return failureTag == null ? VerificationFailureTag.VALIDATION_ERROR : failureTag;
    }

    private boolean isDirectRequest(String content) {
        String normalized = content == null ? "" : content.trim().toLowerCase();
        return normalized.matches("^(hi|hello|hey)[!.? ]*$")
                || normalized.contains("who are you");
    }
}





