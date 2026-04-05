package com.JJLin.aiagent.memory;

import com.JJLin.aiagent.entites.WorkflowMemory;

import java.util.Optional;

public interface WorkflowMemoryStore {

    Optional<WorkflowMemory> find(String sessionId);

    WorkflowMemory save(WorkflowMemory workflowMemory);

    void clear(String sessionId);
}
