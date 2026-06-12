package com.JJLin.aiagent.rag;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.time.LocalDate;
import java.util.List;

@Data
public class KnowledgeDocumentRequest {
    @NotBlank
    private String documentId;
    @NotNull
    private KnowledgeDomain domain;
    private String tenantId;
    @NotBlank
    private String title;
    @NotBlank
    private String source;
    @NotBlank
    private String content;
    private LocalDate validFrom;
    private LocalDate validTo;
    private List<String> tags;
}
