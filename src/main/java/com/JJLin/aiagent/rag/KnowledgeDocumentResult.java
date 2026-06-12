package com.JJLin.aiagent.rag;

import java.time.LocalDate;

public record KnowledgeDocumentResult(
        String documentId,
        KnowledgeDomain domain,
        String tenantId,
        String status,
        int chunkCount,
        LocalDate validFrom,
        LocalDate validTo) {
}
