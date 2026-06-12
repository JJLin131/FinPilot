package com.JJLin.aiagent.service;

import com.JJLin.aiagent.agent.AgentRequestContext;
import com.JJLin.aiagent.agent.ConversationLockService;
import com.JJLin.aiagent.agent.FinanceAssistant;
import com.JJLin.aiagent.api.AgentChatResponse;
import com.JJLin.aiagent.api.FinanceChatRequest;
import org.springframework.stereotype.Service;

import java.util.UUID;

@Service
public class FinanceAgentService {

    private final FinanceAssistant assistant;
    private final ConversationLockService lockService;

    public FinanceAgentService(FinanceAssistant assistant, ConversationLockService lockService) {
        this.assistant = assistant;
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
            String answer = assistant.chat(memoryId, request.getContent());
            return AgentChatResponse.builder().requestId(requestId).domain("FINANCE").status("SUCCEEDED")
                    .answer(answer).evidence(AgentRequestContext.evidence()).build();
        } finally {
            AgentRequestContext.close();
        }
    }
}
