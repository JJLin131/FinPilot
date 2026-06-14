package com.JJLin.aiagent.service;

import com.JJLin.aiagent.agent.ConversationLockService;
import com.JJLin.aiagent.api.FinanceChatRequest;
import com.JJLin.aiagent.route.IntentRouterService;
import com.JJLin.aiagent.route.RouteDecision;
import com.JJLin.aiagent.route.RoutedAgentExecution;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentServiceTest {

    @Test
    void includesTenantInFinanceMemoryId() {
        IntentRouterService routerService = mock(IntentRouterService.class);
        when(routerService.routeAndExecute("finance:tenant-a:u-1:c-1", "summary"))
                .thenReturn(RoutedAgentExecution.builder()
                        .answer("answer")
                        .routeDecision(RouteDecision.builder()
                                .normalizedIntent("GENERAL_FINANCE")
                                .reason("general finance question")
                                .confidence(0.80d)
                                .valid(true)
                                .targetAgent("QueryAgent")
                                .build())
                        .build());
        FinanceChatRequest request = new FinanceChatRequest();
        request.setTenantId("tenant-a");
        request.setUserId("u-1");
        request.setChatId("c-1");
        request.setContent("summary");

        var response = new FinanceAgentService(routerService, new ConversationLockService()).chat(request);

        assertEquals("FINANCE", response.getDomain());
        verify(routerService).routeAndExecute("finance:tenant-a:u-1:c-1", "summary");
    }
}
