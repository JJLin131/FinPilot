package com.JJLin.aiagent.rag;

import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class RagEvaluationService {

    private final RagKnowledgeService knowledgeService;

    public RagEvaluationService(RagKnowledgeService knowledgeService) {
        this.knowledgeService = knowledgeService;
    }

    public RagEvaluationResult evaluate(RagEvaluationRequest request) {
        double recallTotal = 0;
        double precisionTotal = 0;
        double reciprocalRankTotal = 0;
        for (RagEvaluationCase evaluationCase : request.getCases()) {
            List<RagMatch> results = knowledgeService.search(evaluationCase.getDomain(), evaluationCase.getTenantId(),
                    evaluationCase.getQuery(), request.getK());
            long hits = results.stream()
                    .map(RagMatch::documentId)
                    .filter(evaluationCase.getRelevantDocumentIds()::contains)
                    .distinct()
                    .count();
            recallTotal += hits / (double) evaluationCase.getRelevantDocumentIds().size();
            precisionTotal += results.isEmpty() ? 0 : hits / (double) results.size();
            for (int index = 0; index < results.size(); index++) {
                if (evaluationCase.getRelevantDocumentIds().contains(results.get(index).documentId())) {
                    reciprocalRankTotal += 1.0 / (index + 1);
                    break;
                }
            }
        }
        int cases = request.getCases().size();
        return new RagEvaluationResult(
                cases,
                request.getK(),
                recallTotal / cases,
                precisionTotal / cases,
                reciprocalRankTotal / cases);
    }
}
