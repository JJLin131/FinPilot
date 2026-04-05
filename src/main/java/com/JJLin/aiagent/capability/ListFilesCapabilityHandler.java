package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.WorkspaceFileClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public class ListFilesCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final WorkspaceFileClient workspaceFileClient;

    public ListFilesCapabilityHandler(WorkspaceFileClient workspaceFileClient) {
        this.workspaceFileClient = workspaceFileClient;
    }

    @Override
    public String actionType() {
        return "LIST_FILES";
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
                .description("List files under the configured workspace root.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("path")
                                .type("string")
                                .required(false)
                                .description("Relative directory path under workspace root.")
                                .aliases(List.of("directory"))
                                .build(),
                        CapabilityParameter.builder()
                                .name("recursive")
                                .type("boolean")
                                .required(false)
                                .description("Whether to recursively list files.")
                                .aliases(List.of())
                                .build(),
                        CapabilityParameter.builder()
                                .name("maxEntries")
                                .type("integer")
                                .required(false)
                                .description("Maximum number of entries to return.")
                                .aliases(List.of("limit"))
                                .build()))
                .exampleParams(Map.of("path", "src/main/java", "recursive", true, "maxEntries", 50))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        Map<String, Object> normalizedParams = pickKnownParams(actionSpec, Map.of(
                "path", List.of("path", "directory"),
                "recursive", List.of("recursive"),
                "maxEntries", List.of("maxEntries", "limit")));
        String path = normalizedParams.get("path") == null ? "." : String.valueOf(normalizedParams.get("path"));
        boolean recursive = Boolean.parseBoolean(String.valueOf(normalizedParams.getOrDefault("recursive", false)));
        int maxEntries = parseInt(normalizedParams.get("maxEntries"), 100);
        try {
            List<String> files = workspaceFileClient.listFiles(path, recursive, maxEntries);
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("path", path);
            payload.put("files", files);
            payload.put("count", files.size());
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

    private int parseInt(Object value, int defaultValue) {
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(String.valueOf(value));
        } catch (NumberFormatException ex) {
            return defaultValue;
        }
    }
}
