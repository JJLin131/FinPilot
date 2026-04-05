package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract class AbstractHttpCapabilityHandler implements CapabilityHandler {

    protected ExecutionResult missingParam(String actionType, String targetService, String paramName) {
        return ExecutionResult.builder()
                .status(WorkflowTaskStatus.FAILED)
                .actionType(actionType)
                .targetService(targetService)
                .errorCode("MISSING_PARAM")
                .message(paramName + " is required")
                .retryable(Boolean.FALSE)
                .build();
    }

    protected ExecutionResult success(String actionType, String targetService, Object payload, Map<String, Object> evidence) {
        return ExecutionResult.builder()
                .status(WorkflowTaskStatus.SUCCEEDED)
                .actionType(actionType)
                .targetService(targetService)
                .payload(payload)
                .evidence(evidence)
                .message("Capability execution succeeded")
                .retryable(Boolean.FALSE)
                .build();
    }

    protected ExecutionResult failure(String actionType, String targetService, Exception ex, boolean retryable) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("exceptionType", ex.getClass().getName());
        return failure(actionType, targetService, "HTTP_REQUEST_FAILED", ex.getMessage(), evidence, retryable);
    }

    protected ExecutionResult failure(
            String actionType,
            String targetService,
            String errorCode,
            String message,
            Map<String, Object> evidence,
            boolean retryable) {
        return ExecutionResult.builder()
                .status(WorkflowTaskStatus.FAILED)
                .actionType(actionType)
                .targetService(targetService)
                .evidence(evidence)
                .errorCode(errorCode)
                .message(message)
                .retryable(retryable)
                .build();
    }

    protected String stringParam(ActionSpec actionSpec, String key) {
        Object value = actionSpec.getParams() == null ? null : actionSpec.getParams().get(key);
        return value == null ? null : String.valueOf(value);
    }

    protected Map<String, Object> params(ActionSpec actionSpec) {
        return actionSpec.getParams() == null ? Collections.emptyMap() : actionSpec.getParams();
    }

    protected Map<String, Object> pickKnownParams(ActionSpec actionSpec,
                                                  Map<String, List<String>> aliasesByCanonicalName) {
        Map<String, Object> normalized = new LinkedHashMap<>();
        Map<String, Object> source = params(actionSpec);
        aliasesByCanonicalName.forEach((canonicalName, aliases) -> {
            Object value = firstPresent(source, aliases);
            if (value != null) {
                normalized.put(canonicalName, value);
            }
        });
        return normalized;
    }

    protected ExecutionResult invalidParam(String actionType, String targetService, String message) {
        return ExecutionResult.builder()
                .status(WorkflowTaskStatus.FAILED)
                .actionType(actionType)
                .targetService(targetService)
                .errorCode("INVALID_PARAM")
                .message(message)
                .retryable(Boolean.FALSE)
                .build();
    }

    protected Object firstPresent(Map<String, Object> source, List<String> candidateKeys) {
        for (String candidateKey : candidateKeys) {
            if (source.containsKey(candidateKey) && source.get(candidateKey) != null) {
                return source.get(candidateKey);
            }
        }
        return null;
    }

    protected Set<String> providedParamNames(ActionSpec actionSpec) {
        return params(actionSpec).keySet();
    }
}
