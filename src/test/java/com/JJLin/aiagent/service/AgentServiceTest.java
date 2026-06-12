package com.JJLin.aiagent.service;

import com.JJLin.aiagent.agent.ConversationLockService;
import com.JJLin.aiagent.agent.FinanceAssistant;
import com.JJLin.aiagent.api.FinanceChatRequest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentServiceTest {

    @Test
    void includesTenantInFinanceMemoryId() {
        FinanceAssistant assistant = mock(FinanceAssistant.class);
        when(assistant.chat("finance:tenant-a:u-1:c-1", "summary")).thenReturn("answer");
        FinanceChatRequest request = new FinanceChatRequest();
        request.setTenantId("tenant-a");
        request.setUserId("u-1");
        request.setChatId("c-1");
        request.setContent("summary");

        var response = new FinanceAgentService(assistant, new ConversationLockService()).chat(request);

        assertEquals("FINANCE", response.getDomain());
        verify(assistant).chat("finance:tenant-a:u-1:c-1", "summary");
    }
}
