package com.JJLin.aiagent.memory;

import com.JJLin.aiagent.entites.WorkflowMemory;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

public class InMemoryWorkflowMemoryStore implements WorkflowMemoryStore {

    private final Map<String, WorkflowMemory> storage = new ConcurrentHashMap<>();

    @Override
    public Optional<WorkflowMemory> find(String sessionId) {
        return Optional.ofNullable(storage.get(sessionId));
    }

    @Override
    public WorkflowMemory save(WorkflowMemory workflowMemory) {
        storage.put(workflowMemory.getSessionId(), workflowMemory);
        return workflowMemory;
    }

    @Override
    public void clear(String sessionId) {
        storage.remove(sessionId);
    }
}