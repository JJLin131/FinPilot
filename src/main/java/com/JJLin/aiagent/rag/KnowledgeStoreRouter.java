package com.JJLin.aiagent.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

@Component
public class KnowledgeStoreRouter {

    private final EmbeddingStore<TextSegment> financeStore;

    public KnowledgeStoreRouter(@Lazy EmbeddingStore<TextSegment> financeStore) {
        this.financeStore = financeStore;
    }

    public EmbeddingStore<TextSegment> store(KnowledgeDomain domain) {
        if (domain != KnowledgeDomain.FINANCE) {
            throw new IllegalArgumentException("Only the FINANCE knowledge domain is supported.");
        }
        return financeStore;
    }
}
