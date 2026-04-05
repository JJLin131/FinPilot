package com.JJLin.aiagent.entites;

import lombok.Data;

import java.util.Map;

@Data
public class ActionSpec {

    private String taskId;

    private String actionType;

    private String targetService;

    private Map<String, Object> params;

    private String expectedOutput;
}
