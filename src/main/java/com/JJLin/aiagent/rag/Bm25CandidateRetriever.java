package com.JJLin.aiagent.rag;

import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class Bm25CandidateRetriever {

    private final Bm25ChunkIndex index;

    public Bm25CandidateRetriever(Bm25ChunkIndex index) {
        this.index = index;
    }

    public List<RagMatch> retrieve(KnowledgeDomain domain, String query, int limit) {
        return index.search(domain, query, limit);
    }
}
