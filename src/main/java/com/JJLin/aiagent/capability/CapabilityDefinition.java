package com.JJLin.aiagent.capability;

import lombok.Builder;
import lombok.Value;

import java.util.List;
import java.util.Map;

@Value
@Builder
public class CapabilityDefinition {

    String actionType;

    String targetService;

    String description;

    List<CapabilityParameter> parameters;

    Map<String, Object> exampleParams;
}
