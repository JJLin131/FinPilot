package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CrossEncoderRagRerankerTest {

    @Test
    void ordersCandidatesUsingCrossEncoderScores() {
        CrossEncoderScoringClient client = mock(CrossEncoderScoringClient.class);
        when(client.score("query", List.of("first", "second"))).thenReturn(List.of(0.1, 0.9));
        CrossEncoderRagReranker reranker = reranker(client);

        var results = reranker.rerank("query", List.of(match("a", "first", 0.8), match("b", "second", 0.2)), 2);

        assertEquals("b", results.get(0).documentId());
        assertEquals(0.9, results.get(0).score());
    }

    @Test
    void fallsBackToRrfScoresWhenCrossEncoderFails() {
        CrossEncoderScoringClient client = mock(CrossEncoderScoringClient.class);
        when(client.score("query", List.of("first", "second"))).thenThrow(new IllegalStateException("offline"));
        CrossEncoderRagReranker reranker = reranker(client);

        var results = reranker.rerank("query", List.of(match("a", "first", 0.8), match("b", "second", 0.2)), 2);

        assertEquals("a", results.get(0).documentId());
    }

    private CrossEncoderRagReranker reranker(CrossEncoderScoringClient client) {
        return new CrossEncoderRagReranker(client, new ScoreRagReranker(), new KnowledgeProperties());
    }

    private RagMatch match(String documentId, String text, double score) {
        return new RagMatch(documentId, documentId, documentId, text, score);
    }
}
