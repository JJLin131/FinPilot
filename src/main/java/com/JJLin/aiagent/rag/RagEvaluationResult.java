package com.JJLin.aiagent.rag;

public record RagEvaluationResult(
        int cases,
        int k,
        double recallAtK,
        double precisionAtK,
        double meanReciprocalRank) {
}
