package com.JJLin.aiagent.agent;

import com.JJLin.aiagent.api.AgentEvidence;

import java.util.ArrayList;
import java.util.List;

public final class AgentRequestContext {

    private static final ThreadLocal<State> CURRENT = new ThreadLocal<>();

    private AgentRequestContext() {
    }

    public static void open(String requestId, String domain, String tenantId, String userId) {
        CURRENT.set(new State(requestId, domain, tenantId, userId, new ArrayList<>()));
    }

    public static State require() {
        State state = CURRENT.get();
        if (state == null) {
            throw new IllegalStateException("Agent request context is not available.");
        }
        return state;
    }

    public static void addEvidence(AgentEvidence evidence) {
        require().evidence().add(evidence);
    }

    public static List<AgentEvidence> evidence() {
        return List.copyOf(require().evidence());
    }

    public static void close() {
        CURRENT.remove();
    }

    public record State(String requestId, String domain, String tenantId, String userId, List<AgentEvidence> evidence) {
    }
}
