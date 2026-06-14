package com.JJLin.aiagent.api;

import com.JJLin.aiagent.route.RouteDecision;
import lombok.Builder;
import lombok.Value;

import java.util.List;

@Value
@Builder
public class AgentChatResponse {
    String requestId;
    String domain;
    String status;
    String answer;
    List<AgentEvidence> evidence;
    RouteDecision route;
}
