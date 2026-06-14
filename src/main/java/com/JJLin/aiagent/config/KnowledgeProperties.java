package com.JJLin.aiagent.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Data
@ConfigurationProperties(prefix = "ai.knowledge")
public class KnowledgeProperties {
    private String chromaBaseUrl = "http://localhost:8000";
    private String financeCollection = "finance-knowledge-bge-m3-v1";
    private String embeddingBaseUrl = "http://localhost:11434";
    private String embeddingModelName = "bge-m3";
    private int chunkSize = 800;
    private int chunkOverlap = 100;
    private String resourcePattern = "classpath*:*.md";
    private String bm25IndexPath = "./data/lucene/knowledge";
    private int rrfK = 60;
    private String rerankerBaseUrl = "http://localhost:8081";
    private boolean rerankerEnabled = true;
    private int rerankerTimeoutSeconds = 30;
    private String queryRewriterBaseUrl = "http://localhost:11434";
    private String queryRewriterModelName = "qwen2.5:3b";
    private boolean queryRewriterEnabled = true;
    private int queryRewriterTimeoutSeconds = 30;
    private int queryRewriterMaxRewrites = 3;
}
