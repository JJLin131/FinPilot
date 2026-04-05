package com.JJLin.aiagent.entites;

import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import lombok.Builder;
import lombok.Data;

import java.util.Map;

@Data
@Builder
public class ExecutionResult {

    private String taskId;

    private WorkflowTaskStatus status;

    private String actionType;

    private String targetService;

    private Object payload;

    private Map<String, Object> evidence;

    private String errorCode;

    private String message;

    private Boolean retryable;
}
