package com.JJLin.aiagent.rag;

import lombok.Data;

import java.util.List;

@Data
public class CuratedKnowledgeDocument {
    private String title;
    private String normalizedMarkdown;
    private List<String> tags;
}
