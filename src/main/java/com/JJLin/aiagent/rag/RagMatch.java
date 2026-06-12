package com.JJLin.aiagent.rag;

public record RagMatch(
        String documentId,
        String title,
        String source,
        String text,
        double score) {
}
