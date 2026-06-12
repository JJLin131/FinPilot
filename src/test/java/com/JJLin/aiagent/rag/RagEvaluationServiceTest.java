package com.JJLin.aiagent.rag;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RagEvaluationServiceTest {

    @Test
    void calculatesRecallPrecisionAndMrr() {
        RagKnowledgeService knowledgeService = mock(RagKnowledgeService.class);
        when(knowledgeService.search(KnowledgeDomain.FINANCE, "tenant-a", "refund", 3)).thenReturn(List.of(
                new RagMatch("irrelevant", "x", "x", "x", 0.9),
                new RagMatch("relevant", "y", "y", "y", 0.8)));
        RagEvaluationCase evaluationCase = new RagEvaluationCase();
        evaluationCase.setDomain(KnowledgeDomain.FINANCE);
        evaluationCase.setTenantId("tenant-a");
        evaluationCase.setQuery("refund");
        evaluationCase.setRelevantDocumentIds(List.of("relevant"));
        RagEvaluationRequest request = new RagEvaluationRequest();
        request.setK(3);
        request.setCases(List.of(evaluationCase));

        RagEvaluationResult result = new RagEvaluationService(knowledgeService).evaluate(request);

        assertEquals(1.0, result.recallAtK());
        assertEquals(0.5, result.precisionAtK());
        assertEquals(0.5, result.meanReciprocalRank());
    }
}
