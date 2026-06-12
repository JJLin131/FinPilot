package com.JJLin.aiagent.api;

import lombok.Builder;
import lombok.Value;

import java.util.Map;

@Value
@Builder
public class AgentEvidence {
    String toolName;
    String period;
    String source;
    Map<String, Object> summary;
}
