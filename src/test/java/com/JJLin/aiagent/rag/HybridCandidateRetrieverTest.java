package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class HybridCandidateRetrieverTest {

    @Test
    void rrfPromotesCandidateFoundByBothRetrievers() {
        VectorCandidateRetriever vector = mock(VectorCandidateRetriever.class);
        Bm25CandidateRetriever bm25 = mock(Bm25CandidateRetriever.class);
        RagMatch common = match("common", "common");
        when(vector.retrieve(KnowledgeDomain.FINANCE, "q", 5)).thenReturn(List.of(match("vector", "vector"), common));
        when(bm25.retrieve(KnowledgeDomain.FINANCE, "q", 5)).thenReturn(List.of(match("bm25", "bm25"), common));
        KnowledgeProperties properties = new KnowledgeProperties();

        var results = new HybridCandidateRetriever(vector, bm25, properties)
                .retrieve(KnowledgeDomain.FINANCE, "q", 5);

        assertEquals("common", results.get(0).documentId());
    }

    @Test
    void continuesWithBm25WhenVectorRetrievalFails() {
        VectorCandidateRetriever vector = mock(VectorCandidateRetriever.class);
        Bm25CandidateRetriever bm25 = mock(Bm25CandidateRetriever.class);
        when(vector.retrieve(KnowledgeDomain.FINANCE, "q", 5)).thenThrow(new IllegalStateException("chroma offline"));
        when(bm25.retrieve(KnowledgeDomain.FINANCE, "q", 5)).thenReturn(List.of(match("bm25", "bm25")));

        var results = new HybridCandidateRetriever(vector, bm25, new KnowledgeProperties())
                .retrieve(KnowledgeDomain.FINANCE, "q", 5);

        assertEquals("bm25", results.get(0).documentId());
    }

    private RagMatch match(String documentId, String text) {
        return new RagMatch(documentId, documentId, documentId, text, 1);
    }
}
