package com.JJLin.aiagent.controller;

import com.JJLin.aiagent.entites.ChatRequest;
import com.JJLin.aiagent.entites.ChatResponse;
import com.JJLin.aiagent.service.WorkflowOrchestratorService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/workflows")
public class WorkflowController {

    private final WorkflowOrchestratorService workflowOrchestratorService;

    public WorkflowController(WorkflowOrchestratorService workflowOrchestratorService) {
        this.workflowOrchestratorService = workflowOrchestratorService;
    }

    @PostMapping("/chat")
    public ChatResponse chat(@Valid @RequestBody ChatRequest request) {
        return workflowOrchestratorService.chat(request);
    }
}
