package com.JJLin.aiagent.controller;

import com.JJLin.aiagent.api.AgentChatResponse;
import com.JJLin.aiagent.api.FinanceChatRequest;
import com.JJLin.aiagent.service.FinanceAgentService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/finance")
public class FinanceAgentController {

    private final FinanceAgentService agentService;

    public FinanceAgentController(FinanceAgentService agentService) {
        this.agentService = agentService;
    }

    @PostMapping("/chat")
    public AgentChatResponse chat(@Valid @RequestBody FinanceChatRequest request) {
        return agentService.chat(request);
    }
}
