package com.JJLin.aiagent.service;

import com.JJLin.aiagent.capability.CapabilityDefinition;
import com.JJLin.aiagent.capability.CapabilityParameter;
import com.JJLin.aiagent.config.WorkflowProperties;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ChatRequest;
import com.JJLin.aiagent.entites.ChatResponse;
import com.JJLin.aiagent.entites.ExecutionPlan;
import com.JJLin.aiagent.entites.ExecutionResult;
import com.JJLin.aiagent.entites.PlanDraft;
import com.JJLin.aiagent.entites.UserProfile;
import com.JJLin.aiagent.entites.VerificationResult;
import com.JJLin.aiagent.enums.VerificationFailureTag;
import com.JJLin.aiagent.enums.VerificationType;
import com.JJLin.aiagent.enums.WorkflowStage;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import com.JJLin.aiagent.memory.InMemoryWorkflowMemoryStore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Answers;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class WorkflowOrchestratorServiceTest {

    private ChatClient orchestratorAgent;
    private ChatClient planAgent;
    private ChatClient executorAgent;
    private ChatClient verifyAgent;
    private CapabilityDispatcher capabilityDispatcher;
    private WorkflowOrchestratorService service;
    private ChatMemory conversationMemory;
    private UserProfileService userProfileService;
    private UserLongTermMemoryService userLongTermMemoryService;
    private SystemOperationMemoryService systemOperationMemoryService;

    @BeforeEach
    void setUp() {
        orchestratorAgent = mock(ChatClient.class, Answers.RETURNS_DEEP_STUBS);
        planAgent = mock(ChatClient.class, Answers.RETURNS_DEEP_STUBS);
        executorAgent = mock(ChatClient.class, Answers.RETURNS_DEEP_STUBS);
        verifyAgent = mock(ChatClient.class, Answers.RETURNS_DEEP_STUBS);
        capabilityDispatcher = mock(CapabilityDispatcher.class);
        userProfileService = mock(UserProfileService.class);
        userLongTermMemoryService = mock(UserLongTermMemoryService.class);
        systemOperationMemoryService = mock(SystemOperationMemoryService.class);
        conversationMemory = MessageWindowChatMemory.builder()
                .chatMemoryRepository(new InMemoryChatMemoryRepository())
                .maxMessages(20)
                .build();

        WorkflowProperties workflowProperties = new WorkflowProperties();
        workflowProperties.setMaxRetries(1);

        service = new WorkflowOrchestratorService(
                orchestratorAgent,
                planAgent,
                executorAgent,
                verifyAgent,
                capabilityDispatcher,
                workflowProperties,
                conversationMemory,
                new WorkflowMemoryService(new InMemoryWorkflowMemoryStore()),
                new MemoryContextAssembler(),
                userProfileService,
                userLongTermMemoryService,
                systemOperationMemoryService);

        when(capabilityDispatcher.describeCapabilities())
                .thenReturn(Map.of(
                        "SEARCH_PRODUCTS", "product-service",
                        "CHECK_ORDER", "order-service",
                        "RECOMMEND_PRODUCTS", "product-service"));
        when(capabilityDispatcher.definition("SEARCH_PRODUCTS"))
                .thenReturn(Optional.of(capabilityDefinition(
                        "SEARCH_PRODUCTS",
                        "product-service",
                        CapabilityParameter.builder().name("keyword").type("string").required(true).build())));
        when(capabilityDispatcher.definition("RECOMMEND_PRODUCTS"))
                .thenReturn(Optional.of(capabilityDefinition(
                        "RECOMMEND_PRODUCTS",
                        "product-service",
                        CapabilityParameter.builder().name("category").type("string").required(true).build(),
                        CapabilityParameter.builder().name("limit").type("integer").required(false).build())));
        when(capabilityDispatcher.definition("CHECK_ORDER"))
                .thenReturn(Optional.of(capabilityDefinition(
                        "CHECK_ORDER",
                        "order-service",
                        CapabilityParameter.builder().name("orderNo").type("string").required(false).build(),
                        CapabilityParameter.builder().name("userId").type("string").required(true).build())));
        when(orchestratorAgent.prompt().user(anyString()).call().content()).thenReturn("assistant-reply");
        when(userProfileService.getByUserId(anyString())).thenReturn(null);
        when(userLongTermMemoryService.findRelevant(anyString(), anyString(), anyInt())).thenReturn(List.of());
        when(systemOperationMemoryService.findRelevant(anyString(), anyString(), anyInt())).thenReturn(List.of());
    }

    @Test
    void shouldShortCircuitDirectRequests() {
        ChatRequest request = request("hi", "c-1");

        ChatResponse response = service.chat(request);

        assertEquals(WorkflowStage.RESPOND, response.getStage());
        assertEquals("assistant-reply", response.getReply());
        assertTrue(response.getExecutionResults().isEmpty());
        assertEquals(List.of(WorkflowStage.INTAKE, WorkflowStage.RESPOND),
                response.getWorkflowState().getStageHistory());
        verify(planAgent, never()).prompt();
        verify(executorAgent, never()).prompt();
        verify(verifyAgent, never()).prompt();
    }

    @Test
    void shouldStopAtFailsafeWhenExecutorNeedsMoreBusinessInfo() {
        PlanDraft planDraft = intentPlan("Check order after identifying the order reference");
        planDraft.setMissingInfo(List.of("orderNo"));

        ExecutionPlan executionPlan = new ExecutionPlan();
        executionPlan.setGoal("check order");
        executionPlan.setBlocked(Boolean.TRUE);
        executionPlan.setMissingInfo(List.of("orderNo"));
        executionPlan.setBlockReason("Need the user's order reference");

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(passVerification(VerificationType.PLAN));

        ChatResponse response = service.chat(request("please query my order", "c-1"));

        assertEquals(WorkflowStage.FAILSAFE, response.getStage());
        assertEquals(VerificationFailureTag.MISSING_DATA, response.getWorkflowState().getFailureReason());
        assertEquals(VerificationType.EXECUTION, response.getVerificationResult().getVerificationType());
        assertTrue(response.getReply().contains("orderNo"));
        assertNotNull(response.getMemorySummary());
        assertTrue(response.getMemorySummary().contains("lastFailureTag=MISSING_DATA"));
        verify(capabilityDispatcher, never()).dispatch(any());
        verify(systemOperationMemoryService).record(anyString(), anyString(), anyString());
    }

    @Test
    void shouldRejectUnsupportedActionsBeforeDispatchWhenExecutorChoosesUnknownHandler() {
        PlanDraft planDraft = intentPlan("Use an unsupported action");
        ExecutionPlan executionPlan = executionPlanStep("UNKNOWN_ACTION", "product-service", Map.of("keyword", "shoe"));
        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(passVerification(VerificationType.PLAN));
        when(capabilityDispatcher.supports("UNKNOWN_ACTION")).thenReturn(false);
        when(capabilityDispatcher.definition("UNKNOWN_ACTION")).thenReturn(Optional.empty());

        ChatResponse response = service.chat(request("do something unsupported", "c-1"));

        assertEquals(WorkflowStage.FAILSAFE, response.getStage());
        assertEquals(VerificationFailureTag.UNSUPPORTED_ACTION, response.getVerificationResult().getFailureTag());
        verify(capabilityDispatcher, never()).dispatch(any());
    }

    @Test
    void shouldExecuteVerifiedPlansAndReturnStructuredState() {
        PlanDraft planDraft = intentPlan("Search for a product");
        ExecutionPlan executionPlan = executionPlanStep("SEARCH_PRODUCTS", "product-service", Map.of("keyword", "shoe"));
        VerificationResult planVerification = passVerification(VerificationType.PLAN);
        VerificationResult executionVerification = passVerification(VerificationType.EXECUTION);
        ExecutionResult executionResult = ExecutionResult.builder()
                .status(WorkflowTaskStatus.SUCCEEDED)
                .actionType("SEARCH_PRODUCTS")
                .targetService("product-service")
                .payload(Map.of("items", List.of("shoe")))
                .retryable(Boolean.FALSE)
                .build();

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(planVerification, executionVerification);
        when(capabilityDispatcher.supports("SEARCH_PRODUCTS")).thenReturn(true);
        when(capabilityDispatcher.validateAction(any())).thenReturn(List.of());
        when(capabilityDispatcher.dispatch(any())).thenAnswer(invocation -> {
            ActionSpec actionSpec = invocation.getArgument(0, ActionSpec.class);
            executionResult.setTaskId(actionSpec.getTaskId());
            return executionResult;
        });

        ChatResponse response = service.chat(request("find me shoes", "c-1"));

        assertEquals(WorkflowStage.RESPOND, response.getStage());
        assertEquals("assistant-reply", response.getReply());
        assertEquals(1, response.getExecutionResults().size());
        assertNotNull(response.getExecutionPlan());
        assertEquals("SEARCH_PRODUCTS", response.getExecutionPlan().getSteps().get(0).getActionType());
        assertEquals("shoe", response.getExecutionPlan().getSteps().get(0).getParams().get("keyword"));
        assertNotNull(response.getExecutionResults().get(0).getTaskId());
        assertTrue(response.getWorkflowState().getStageHistory().contains(WorkflowStage.PLAN_VERIFY));
        assertTrue(response.getWorkflowState().getStageHistory().contains(WorkflowStage.EXECUTION_VERIFY));
        assertNotNull(response.getMemorySummary());
        assertTrue(response.getMemorySummary().contains("latestPlan=Search for a product"));
    }

    @Test
    void shouldFailWhenExecutionResultsContainFailedSteps() {
        PlanDraft planDraft = intentPlan("Search for a product");
        ExecutionPlan executionPlan = executionPlanStep("SEARCH_PRODUCTS", "product-service", Map.of("keyword", "shoe"));
        VerificationResult planVerification = passVerification(VerificationType.PLAN);
        VerificationResult executionVerification = passVerification(VerificationType.EXECUTION);
        ExecutionResult failedExecution = ExecutionResult.builder()
                .status(WorkflowTaskStatus.FAILED)
                .actionType("SEARCH_PRODUCTS")
                .targetService("product-service")
                .errorCode("HTTP_REQUEST_FAILED")
                .message("backend timeout")
                .retryable(Boolean.TRUE)
                .build();

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(planVerification, executionVerification);
        when(capabilityDispatcher.supports("SEARCH_PRODUCTS")).thenReturn(true);
        when(capabilityDispatcher.validateAction(any())).thenReturn(List.of());
        when(capabilityDispatcher.dispatch(any())).thenAnswer(invocation -> {
            ActionSpec actionSpec = invocation.getArgument(0, ActionSpec.class);
            failedExecution.setTaskId(actionSpec.getTaskId());
            return failedExecution;
        });

        ChatResponse response = service.chat(request("find me shoes", "c-1"));

        assertEquals(WorkflowStage.FAILSAFE, response.getStage());
        assertEquals(VerificationFailureTag.TOOL_FAILURE, response.getVerificationResult().getFailureTag());
        assertFalse(response.getWorkflowState().getRequiredFixes().isEmpty());
        assertNotNull(response.getMemorySummary());
        assertTrue(response.getMemorySummary().contains("lastFailureTag=TOOL_FAILURE"));
    }

    @Test
    void shouldInjectPreviousTurnWorkflowStateIntoSubsequentPlanPrompt() {
        UserProfile userProfile = new UserProfile();
        userProfile.setUserId("u-1");
        userProfile.setOccupation("teacher");
        when(userProfileService.getByUserId("u-1")).thenReturn(userProfile);

        PlanDraft firstPlan = intentPlan("User asked for shoes");
        firstPlan.setGoal("first intent");
        firstPlan.setContextSummary("User asked for shoes");
        firstPlan.setAssumptions(List.of("user wants shoes"));

        PlanDraft secondPlan = intentPlan("User asks about an order after product search");
        secondPlan.setGoal("second intent");
        secondPlan.setContextSummary("User asks about an order after product search");
        secondPlan.setAssumptions(List.of("order number is o-1"));

        ExecutionPlan firstExecutionPlan = executionPlanStep("SEARCH_PRODUCTS", "product-service", Map.of("keyword", "shoe"));
        ExecutionPlan secondExecutionPlan = executionPlanStep("CHECK_ORDER", "order-service", Map.of("orderId", "o-1"));

        VerificationResult passPlan = passVerification(VerificationType.PLAN);
        VerificationResult passExecution = passVerification(VerificationType.EXECUTION);
        ExecutionResult success = ExecutionResult.builder()
                .status(WorkflowTaskStatus.SUCCEEDED)
                .actionType("SEARCH_PRODUCTS")
                .targetService("product-service")
                .payload(Map.of("ok", true))
                .retryable(Boolean.FALSE)
                .build();

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(firstPlan, secondPlan);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class))
                .thenReturn(firstExecutionPlan, secondExecutionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(passPlan, passExecution, passPlan, passExecution);
        when(capabilityDispatcher.supports("SEARCH_PRODUCTS")).thenReturn(true);
        when(capabilityDispatcher.supports("CHECK_ORDER")).thenReturn(true);
        when(capabilityDispatcher.validateAction(any())).thenReturn(List.of());
        when(capabilityDispatcher.dispatch(any())).thenAnswer(invocation -> {
            ActionSpec actionSpec = invocation.getArgument(0, ActionSpec.class);
            success.setTaskId(actionSpec.getTaskId());
            success.setActionType(actionSpec.getActionType());
            success.setTargetService(actionSpec.getTargetService());
            return success;
        });

        ChatResponse firstResponse = service.chat(request("find me shoes", "memory-chat"));
        ChatResponse secondResponse = service.chat(request("check order o-1", "memory-chat"));

        verify(userProfileService, org.mockito.Mockito.times(2)).updateFromRequestAsync(any(ChatRequest.class), any());
        verify(userLongTermMemoryService, org.mockito.Mockito.times(2)).extractFromRequestAsync(any(ChatRequest.class), any());
        assertTrue(firstResponse.getMemorySummary().contains("resolvedFacts=user wants shoes"));
        assertTrue(secondResponse.getMemorySummary().contains("resolvedFacts=user wants shoes; order number is o-1"));
        assertTrue(secondResponse.getMemorySummary().contains("latestPlan=User asks about an order after product search"));
    }

    @Test
    void shouldAllowIntentOnlyPlanWithoutCapabilityValidationDuringPlanReview() {
        PlanDraft planDraft = intentPlan("Collect preference information and recommend products");
        planDraft.getSteps().get(0).setActionType("SEARCH_PRODUCTS");
        planDraft.getSteps().get(0).setTargetService("product-service");
        planDraft.getSteps().get(0).setParams(Map.of("keyword", "shoe"));
        ExecutionPlan executionPlan = executionPlanStep("RECOMMEND_PRODUCTS", "product-service", Map.of("category", "running shoes"));
        VerificationResult planVerification = passVerification(VerificationType.PLAN);
        VerificationResult executionVerification = passVerification(VerificationType.EXECUTION);
        ExecutionResult executionResult = ExecutionResult.builder()
                .status(WorkflowTaskStatus.SUCCEEDED)
                .actionType("RECOMMEND_PRODUCTS")
                .targetService("product-service")
                .payload(Map.of("items", List.of("shoe")))
                .retryable(Boolean.FALSE)
                .build();

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(planVerification, executionVerification);
        when(capabilityDispatcher.supports("RECOMMEND_PRODUCTS")).thenReturn(true);
        when(capabilityDispatcher.validateAction(any())).thenReturn(List.of());
        when(capabilityDispatcher.dispatch(any())).thenReturn(executionResult);

        ChatResponse response = service.chat(request("recommend me some products", "c-2"));

        assertEquals(WorkflowStage.RESPOND, response.getStage());
        assertEquals(null, response.getPlanDraft().getSteps().get(0).getActionType());
        assertEquals(null, response.getPlanDraft().getSteps().get(0).getTargetService());
        assertEquals(null, response.getPlanDraft().getSteps().get(0).getParams());
    }

    @Test
    void shouldNotFailPlanReviewOnlyBecausePlanContainsMissingInfo() {
        PlanDraft planDraft = intentPlan("Recommend products after clarifying preferences");
        planDraft.setMissingInfo(List.of("budget", "style"));

        ExecutionPlan executionPlan = new ExecutionPlan();
        executionPlan.setGoal("recommend");
        executionPlan.setBlocked(Boolean.TRUE);
        executionPlan.setMissingInfo(List.of("budget", "style"));
        executionPlan.setBlockReason("Need shopping preferences");

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(passVerification(VerificationType.PLAN));

        ChatResponse response = service.chat(request("recommend shoes for me", "c-4"));

        assertEquals(WorkflowStage.FAILSAFE, response.getStage());
        assertEquals(VerificationType.EXECUTION, response.getVerificationResult().getVerificationType());
        assertEquals(VerificationFailureTag.MISSING_DATA, response.getVerificationResult().getFailureTag());
    }

    @Test
    void shouldIgnoreOptionalMissingInfoWhenRequiredParamsAreAlreadyPresent() {
        PlanDraft planDraft = intentPlan("Recommend running shoes within budget");
        planDraft.setMissingInfo(List.of("size", "color", "brand"));

        ExecutionPlan executionPlan = new ExecutionPlan();
        executionPlan.setGoal("recommend running shoes");
        executionPlan.setSteps(List.of(executionStep(
                "RECOMMEND_PRODUCTS",
                "product-service",
                Map.of("category", "running shoes", "limit", 5))));
        executionPlan.setBlocked(Boolean.TRUE);
        executionPlan.setMissingInfo(List.of("size", "color", "brand"));
        executionPlan.setBlockReason("Need more preferences");

        VerificationResult planVerification = passVerification(VerificationType.PLAN);
        VerificationResult executionVerification = passVerification(VerificationType.EXECUTION);
        ExecutionResult executionResult = ExecutionResult.builder()
                .status(WorkflowTaskStatus.SUCCEEDED)
                .actionType("RECOMMEND_PRODUCTS")
                .targetService("product-service")
                .payload(Map.of("items", List.of("shoe")))
                .retryable(Boolean.FALSE)
                .build();

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(planVerification, executionVerification);
        when(capabilityDispatcher.supports("RECOMMEND_PRODUCTS")).thenReturn(true);
        when(capabilityDispatcher.validateAction(any())).thenReturn(List.of());
        when(capabilityDispatcher.dispatch(any())).thenReturn(executionResult);

        ChatResponse response = service.chat(request("I have 500 yuan and want running shoes", "c-5"));

        assertEquals(WorkflowStage.RESPOND, response.getStage());
        assertEquals("RECOMMEND_PRODUCTS", response.getExecutionResults().get(0).getActionType());
        assertEquals("RECOMMEND_PRODUCTS", response.getExecutionPlan().getSteps().get(0).getActionType());
        assertEquals("running shoes", response.getExecutionPlan().getSteps().get(0).getParams().get("category"));
        assertFalse(Boolean.TRUE.equals(response.getExecutionPlan().getBlocked()));
        assertTrue(response.getExecutionPlan().getMissingInfo().isEmpty());
    }

    @Test
    void shouldBindExecutionParamsFromExistingUserSemantics() {
        PlanDraft planDraft = intentPlan("Recommend running shoes within budget");

        ExecutionPlan firstExecutionPlan = new ExecutionPlan();
        firstExecutionPlan.setGoal("recommend running shoes");
        firstExecutionPlan.setSteps(List.of(executionStep(
                "RECOMMEND_PRODUCTS",
                "product-service",
                Map.of())));
        firstExecutionPlan.setBlocked(Boolean.TRUE);
        firstExecutionPlan.setMissingInfo(List.of("Required parameter 'category'"));
        firstExecutionPlan.setBlockReason("Missing category");

        ExecutionPlan repairedExecutionPlan = new ExecutionPlan();
        repairedExecutionPlan.setGoal("recommend running shoes");
        repairedExecutionPlan.setSteps(List.of(executionStep(
                "RECOMMEND_PRODUCTS",
                "product-service",
                Map.of("category", "running shoes", "limit", 5))));
        repairedExecutionPlan.setBlocked(Boolean.FALSE);
        repairedExecutionPlan.setMissingInfo(List.of());

        VerificationResult planVerification = passVerification(VerificationType.PLAN);
        VerificationResult executionVerification = passVerification(VerificationType.EXECUTION);
        ExecutionResult executionResult = ExecutionResult.builder()
                .status(WorkflowTaskStatus.SUCCEEDED)
                .actionType("RECOMMEND_PRODUCTS")
                .targetService("product-service")
                .payload(Map.of("items", List.of("shoe")))
                .retryable(Boolean.FALSE)
                .build();

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class))
                .thenReturn(firstExecutionPlan, repairedExecutionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(planVerification, executionVerification);
        when(capabilityDispatcher.supports("RECOMMEND_PRODUCTS")).thenReturn(true);
        when(capabilityDispatcher.validateAction(any())).thenReturn(List.of());
        when(capabilityDispatcher.dispatch(any())).thenReturn(executionResult);

        ChatResponse response = service.chat(request("I only have 500 yuan and want running shoes", "c-6"));

        assertEquals(WorkflowStage.RESPOND, response.getStage());
        assertEquals("running shoes", response.getExecutionPlan().getSteps().get(0).getParams().get("category"));
        assertFalse(Boolean.TRUE.equals(response.getExecutionPlan().getBlocked()));
    }

    @Test
    void shouldReturnOnlyRequiredMissingInfoWhenExecutorOverAsks() {
        PlanDraft planDraft = intentPlan("Recommend products based on user preferences");
        ExecutionPlan executionPlan = new ExecutionPlan();
        executionPlan.setGoal("recommend");
        executionPlan.setSteps(List.of(executionStep(
                "RECOMMEND_PRODUCTS",
                "product-service",
                Map.of("limit", 5))));
        executionPlan.setBlocked(Boolean.TRUE);
        executionPlan.setMissingInfo(List.of("budget", "brand", "category", "color"));
        executionPlan.setBlockReason("Need core shopping preferences");

        when(planAgent.prompt().user(anyString()).call().entity(PlanDraft.class)).thenReturn(planDraft);
        when(executorAgent.prompt().user(anyString()).call().entity(ExecutionPlan.class)).thenReturn(executionPlan);
        when(verifyAgent.prompt().user(anyString()).call().entity(VerificationResult.class))
                .thenReturn(passVerification(VerificationType.PLAN));

        ChatResponse response = service.chat(request("recommend me something", "c-3"));

        assertEquals(WorkflowStage.FAILSAFE, response.getStage());
        assertEquals(VerificationFailureTag.MISSING_DATA, response.getWorkflowState().getFailureReason());
        assertEquals(List.of("Required parameter 'category'"), response.getExecutionPlan().getMissingInfo());
        assertTrue(response.getReply().contains("category"));
        verify(capabilityDispatcher, never()).dispatch(any());
    }

    private ChatRequest request(String content, String chatId) {
        ChatRequest request = new ChatRequest();
        request.setUserId("u-1");
        request.setChatId(chatId);
        request.setContent(content);
        return request;
    }

    private PlanDraft intentPlan(String contextSummary) {
        PlanDraft planDraft = new PlanDraft();
        planDraft.setGoal("find product");
        planDraft.setPlanType("EXECUTION");
        planDraft.setNeedsExecution(Boolean.TRUE);
        planDraft.setContextSummary(contextSummary);

        PlanDraft.PlanStep step = new PlanDraft.PlanStep();
        step.setTitle("understand request");
        step.setDescription("Gather the information needed to satisfy the shopping request.");
        step.setPrerequisites(List.of("Understand the user's shopping goal"));
        step.setSuccessCriteria(List.of("The next step has enough business context to proceed"));
        step.setExpectedOutput("resolved recommendation inputs");
        planDraft.setSteps(List.of(step));
        return planDraft;
    }

    private ExecutionPlan executionPlanStep(String actionType, String targetService, Map<String, Object> params) {
        ExecutionPlan executionPlan = new ExecutionPlan();
        executionPlan.setGoal("execute");
        executionPlan.setBlocked(Boolean.FALSE);
        executionPlan.setSteps(List.of(executionStep(actionType, targetService, params)));
        return executionPlan;
    }

    private ExecutionPlan.ExecutionStep executionStep(String actionType, String targetService, Map<String, Object> params) {
        ExecutionPlan.ExecutionStep step = new ExecutionPlan.ExecutionStep();
        step.setReason("Chosen by executor");
        step.setActionType(actionType);
        step.setTargetService(targetService);
        step.setParams(params);
        step.setExpectedOutput("result");
        return step;
    }

    private CapabilityDefinition capabilityDefinition(
            String actionType,
            String targetService,
            CapabilityParameter... parameters) {
        return CapabilityDefinition.builder()
                .actionType(actionType)
                .targetService(targetService)
                .description("test capability")
                .parameters(List.of(parameters))
                .exampleParams(Map.of())
                .build();
    }

    private VerificationResult passVerification(VerificationType verificationType) {
        VerificationResult verificationResult = new VerificationResult();
        verificationResult.setPass(Boolean.TRUE);
        verificationResult.setVerificationType(verificationType);
        verificationResult.setFailureTag(VerificationFailureTag.NONE);
        verificationResult.setRequiredFixes(List.of());
        verificationResult.setFindings(List.of());
        verificationResult.setSummary("ok");
        return verificationResult;
    }
}




