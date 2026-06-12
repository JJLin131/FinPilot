package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.WorkspaceFileClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ReadFileCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final WorkspaceFileClient workspaceFileClient;

    public ReadFileCapabilityHandler(WorkspaceFileClient workspaceFileClient) {
        this.workspaceFileClient = workspaceFileClient;
    }

    @Override
    public String actionType() {
        return "READ_FILE";
    }

    @Override
    public String targetService() {
        return "local-workspace";
    }

    @Override
    public CapabilityDefinition definition() {
        return CapabilityDefinition.builder()
                .actionType(actionType())
                .targetService(targetService())
                .description("Read a text file under the configured workspace root.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("path")
                                .type("string")
                                .required(true)
                                .description("Relative file path under workspace root.")
                                .aliases(List.of("filePath"))
                                .build()))
                .exampleParams(Map.of("path", "README.md"))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        Map<String, Object> normalizedParams = pickKnownParams(actionSpec, Map.of(
                "path", List.of("path", "filePath")));
        String path = normalizedParams.get("path") == null ? null : String.valueOf(normalizedParams.get("path"));
        if (path == null || path.isBlank()) {
            return missingParam(actionType(), targetService(), "path");
        }
        try {
            String content = workspaceFileClient.readFile(path);
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("path", path);
            payload.put("content", content);
            payload.put("truncated", false);
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("workspaceRoot", workspaceFileClient.workspaceRoot().toString());
            return success(actionType(), targetService(), payload, evidence);
        } catch (Exception ex) {
            return localFailure(ex);
        }
    }

    private ExecutionResult localFailure(Exception ex) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("exceptionType", ex.getClass().getName());
        return failure(actionType(), targetService(), "LOCAL_IO_ERROR", ex.getMessage(), evidence, false);
    }
}
