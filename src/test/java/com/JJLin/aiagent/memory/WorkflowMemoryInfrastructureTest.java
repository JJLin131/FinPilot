package com.JJLin.aiagent.memory;

import com.JJLin.aiagent.entites.ChatRequest;
import com.JJLin.aiagent.entites.PlanDraft;
import com.JJLin.aiagent.entites.WorkflowMemory;
import com.JJLin.aiagent.entites.WorkflowState;
import com.JJLin.aiagent.enums.VerificationFailureTag;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import com.JJLin.aiagent.service.WorkflowMemoryService;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkflowMemoryInfrastructureTest {

    @Test
    void shouldKeepOnlyRecentTenRoundsInConversationMemory() {
        ChatMemory chatMemory = MessageWindowChatMemory.builder()
                .chatMemoryRepository(new InMemoryChatMemoryRepository())
                .maxMessages(20)
                .build();

        for (int i = 1; i <= 12; i++) {
            chatMemory.add("chat-1", List.of(
                    new UserMessage("user-" + i),
                    new AssistantMessage("assistant-" + i)));
        }

        List<?> messages = chatMemory.get("chat-1");
        assertEquals(20, messages.size());
        assertTrue(messages.toString().contains("user-12"));
        assertTrue(messages.toString().contains("assistant-12"));
        assertTrue(messages.toString().contains("user-3"));
    }

    @Test
    void shouldIsolateWorkflowMemoryAcrossSessions() {
        WorkflowMemoryService workflowMemoryService = new WorkflowMemoryService(new InMemoryWorkflowMemoryStore());

        PlanDraft firstPlan = new PlanDraft();
        firstPlan.setContextSummary("search products");
        firstPlan.setAssumptions(List.of("user wants shoes"));

        PlanDraft secondPlan = new PlanDraft();
        secondPlan.setContextSummary("query order");
        secondPlan.setAssumptions(List.of("user provided order number"));

        workflowMemoryService.updateAfterPlan("chat-a", request("chat-a", "first"), new WorkflowState(), firstPlan);
        workflowMemoryService.updateAfterPlan("chat-b", request("chat-b", "second"), new WorkflowState(), secondPlan);

        WorkflowMemory memoryA = workflowMemoryService.getOrCreate("chat-a");
        WorkflowMemory memoryB = workflowMemoryService.getOrCreate("chat-b");

        assertTrue(memoryA.getResolvedFacts().contains("user wants shoes"));
        assertTrue(memoryB.getResolvedFacts().contains("user provided order number"));
    }

    @Test
    void shouldPersistOnlyTaskStateOnFailedTurn() {
        WorkflowMemoryService workflowMemoryService = new WorkflowMemoryService(new InMemoryWorkflowMemoryStore());

        WorkflowState state = new WorkflowState();
        state.setCurrentTaskStatus(WorkflowTaskStatus.FAILED);
        state.setFailureReason(VerificationFailureTag.MISSING_DATA);
        state.setRequiredFixes(List.of("Ask the user for orderNo."));

        WorkflowMemory memory = workflowMemoryService.finalizeTurn(
                "chat-1",
                request("chat-1", "find order"),
                "need order number",
                state,
                List.of(new UserMessage("find order"), new AssistantMessage("need order number")));

        assertEquals(VerificationFailureTag.MISSING_DATA, memory.getLastFailureTag());
        assertEquals(List.of("Ask the user for orderNo."), memory.getLastRequiredFixes());
        assertTrue(memory.getOpenQuestions().contains("Ask the user for orderNo."));
    }

    private ChatRequest request(String chatId, String content) {
        ChatRequest request = new ChatRequest();
        request.setUserId("u-1");
        request.setChatId(chatId);
        request.setContent(content);
        return request;
    }
}