package com.JJLin.aiagent.rag;

import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;

@Component
public class ScoreRagReranker implements RagReranker {
    @Override
    public List<RagMatch> rerank(String query, List<RagMatch> candidates, int limit) {
        return candidates.stream()
                .sorted(Comparator.comparingDouble(RagMatch::score).reversed())
                .limit(limit)
                .toList();
    }
}
