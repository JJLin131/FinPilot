package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.WorkspaceFileClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public class WriteFileCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final WorkspaceFileClient workspaceFileClient;

    public WriteFileCapabilityHandler(WorkspaceFileClient workspaceFileClient) {
        this.workspaceFileClient = workspaceFileClient;
    }

    @Override
    public String actionType() {
        return "WRITE_FILE";
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
                .description("Write a text file under the configured workspace root.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("path")
                                .type("string")
                                .required(true)
                                .description("Relative file path under workspace root.")
                                .aliases(List.of("filePath"))
                                .build(),
                        CapabilityParameter.builder()
                                .name("content")
                                .type("string")
                                .required(true)
                                .description("Text content to write.")
                                .aliases(List.of())
                                .build(),
                        CapabilityParameter.builder()
                                .name("append")
                                .type("boolean")
                                .required(false)
                                .description("Whether to append instead of overwrite.")
                                .aliases(List.of())
                                .build()))
                .exampleParams(Map.of("path", "notes/output.txt", "content", "hello", "append", false))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        Map<String, Object> normalizedParams = pickKnownParams(actionSpec, Map.of(
                "path", List.of("path", "filePath"),
                "content", List.of("content"),
                "append", List.of("append")));
        String path = normalizedParams.get("path") == null ? null : String.valueOf(normalizedParams.get("path"));
        String content = normalizedParams.get("content") == null ? null : String.valueOf(normalizedParams.get("content"));
        boolean append = Boolean.parseBoolean(String.valueOf(normalizedParams.getOrDefault("append", false)));
        if (path == null || path.isBlank()) {
            return missingParam(actionType(), targetService(), "path");
        }
        if (content == null) {
            return missingParam(actionType(), targetService(), "content");
        }
        try {
            workspaceFileClient.writeFile(path, content, append);
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("path", path);
            payload.put("append", append);
            payload.put("writtenChars", content.length());
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
