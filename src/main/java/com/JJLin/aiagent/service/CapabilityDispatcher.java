package com.JJLin.aiagent.service;

import com.JJLin.aiagent.capability.CapabilityDefinition;
import com.JJLin.aiagent.capability.CapabilityRegistry;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
public class CapabilityDispatcher {

    private final CapabilityRegistry capabilityRegistry;

    public CapabilityDispatcher(CapabilityRegistry capabilityRegistry) {
        this.capabilityRegistry = capabilityRegistry;
    }

    public ExecutionResult dispatch(ActionSpec actionSpec) {
        ExecutionResult result = capabilityRegistry.find(actionSpec.getActionType())
                .map(handler -> handler.execute(actionSpec))
                .orElseGet(() -> unsupported(actionSpec));
        if (result.getTaskId() == null) {
            result.setTaskId(actionSpec.getTaskId());
        }
        return result;
    }

    public Map<String, Object> describeCapabilities() {
        return capabilityRegistry.describe();
    }

    public List<String> validateAction(ActionSpec actionSpec) {
        return capabilityRegistry.validateAction(
                actionSpec.getActionType(),
                actionSpec.getTargetService(),
                actionSpec.getParams());
    }

    public Optional<CapabilityDefinition> definition(String actionType) {
        return capabilityRegistry.definition(actionType);
    }

    //通过 actionType 判断是否有此 action
    public boolean supports(String actionType) {
        return capabilityRegistry.find(actionType).isPresent();
    }

    private ExecutionResult unsupported(ActionSpec actionSpec) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("knownCapabilities", capabilityRegistry.describe());
        return ExecutionResult.builder()
                .taskId(actionSpec.getTaskId())
                .status(WorkflowTaskStatus.FAILED)
                .actionType(actionSpec.getActionType())
                .targetService(actionSpec.getTargetService())
                .evidence(evidence)
                .errorCode("UNSUPPORTED_ACTION")
                .message("No capability registered for actionType=" + actionSpec.getActionType())
                .retryable(Boolean.FALSE)
                .build();
    }
}
