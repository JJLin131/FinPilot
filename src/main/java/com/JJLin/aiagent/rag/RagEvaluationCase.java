package com.JJLin.aiagent.rag;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Data
public class RagEvaluationCase {
    @NotNull
    private KnowledgeDomain domain;
    private String tenantId;
    @NotBlank
    private String query;
    @NotEmpty
    private List<String> relevantDocumentIds;
}
