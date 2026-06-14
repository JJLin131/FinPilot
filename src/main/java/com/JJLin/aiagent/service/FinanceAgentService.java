package com.JJLin.aiagent.service;

import com.JJLin.aiagent.agent.AgentRequestContext;
import com.JJLin.aiagent.agent.ConversationLockService;
import com.JJLin.aiagent.api.AgentChatResponse;
import com.JJLin.aiagent.api.FinanceChatRequest;
import com.JJLin.aiagent.route.IntentRouterService;
import com.JJLin.aiagent.route.RoutedAgentExecution;
import org.springframework.stereotype.Service;

import java.util.UUID;

@Service
public class FinanceAgentService {

    private final IntentRouterService intentRouterService;
    private final ConversationLockService lockService;

    public FinanceAgentService(IntentRouterService intentRouterService, ConversationLockService lockService) {
        this.intentRouterService = intentRouterService;
        this.lockService = lockService;
    }

    public AgentChatResponse chat(FinanceChatRequest request) {
        String memoryId = "finance:" + request.getTenantId() + ":" + request.getUserId() + ":" + request.getChatId();
        return lockService.execute(memoryId, () -> execute(request, memoryId));
    }

    private AgentChatResponse execute(FinanceChatRequest request, String memoryId) {
        String requestId = UUID.randomUUID().toString();
        AgentRequestContext.open(requestId, "FINANCE", request.getTenantId(), request.getUserId());
        try {
            RoutedAgentExecution execution = intentRouterService.routeAndExecute(memoryId, request.getContent());
            return AgentChatResponse.builder().requestId(requestId).domain("FINANCE").status("SUCCEEDED")
                    .answer(execution.getAnswer())
                    .evidence(AgentRequestContext.evidence())
                    .route(execution.getRouteDecision())
                    .build();
        } finally {
            AgentRequestContext.close();
        }
    }
}
