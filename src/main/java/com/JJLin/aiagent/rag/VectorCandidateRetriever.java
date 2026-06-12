package com.JJLin.aiagent.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class VectorCandidateRetriever implements RagCandidateRetriever {

    private final EmbeddingModel embeddingModel;
    private final KnowledgeStoreRouter storeRouter;

    public VectorCandidateRetriever(EmbeddingModel knowledgeEmbeddingModel, KnowledgeStoreRouter storeRouter) {
        this.embeddingModel = knowledgeEmbeddingModel;
        this.storeRouter = storeRouter;
    }

    @Override
    public List<RagMatch> retrieve(KnowledgeDomain domain, String query, int limit) {
        var queryEmbedding = embeddingModel.embed(query).content();
        return storeRouter.store(domain).search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(limit)
                        .minScore(0.45)
                        .build())
                .matches().stream()
                .map(this::toMatch)
                .toList();
    }

    private RagMatch toMatch(dev.langchain4j.store.embedding.EmbeddingMatch<TextSegment> match) {
        TextSegment segment = match.embedded();
        return new RagMatch(segment.metadata().getString("documentId"), segment.metadata().getString("title"),
                segment.metadata().getString("source"), segment.text(), match.score());
    }
}
