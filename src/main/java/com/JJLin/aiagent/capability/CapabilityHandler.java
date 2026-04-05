package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;

public interface CapabilityHandler {

    String actionType();

    String targetService();

    CapabilityDefinition definition();

    ExecutionResult execute(ActionSpec actionSpec);
}
