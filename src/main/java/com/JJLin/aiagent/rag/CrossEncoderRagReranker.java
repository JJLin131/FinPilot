package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Primary
@Component
public class CrossEncoderRagReranker implements RagReranker {

    private static final Logger log = LoggerFactory.getLogger(CrossEncoderRagReranker.class);

    private final CrossEncoderScoringClient scoringClient;
    private final ScoreRagReranker fallback;
    private final boolean enabled;

    public CrossEncoderRagReranker(
            CrossEncoderScoringClient scoringClient,
            ScoreRagReranker fallback,
            KnowledgeProperties properties) {
        this.scoringClient = scoringClient;
        this.fallback = fallback;
        this.enabled = properties.isRerankerEnabled();
    }

    @Override
    public List<RagMatch> rerank(String query, List<RagMatch> candidates, int limit) {
        if (!enabled || candidates.isEmpty()) {
            return fallback.rerank(query, candidates, limit);
        }
        try {
            List<Double> scores = scoringClient.score(query, candidates.stream().map(RagMatch::text).toList());
            List<RagMatch> reranked = new ArrayList<>();
            for (int index = 0; index < candidates.size(); index++) {
                RagMatch candidate = candidates.get(index);
                reranked.add(new RagMatch(candidate.documentId(), candidate.title(), candidate.source(),
                        candidate.text(), scores.get(index)));
            }
            return reranked.stream()
                    .sorted(Comparator.comparingDouble(RagMatch::score).reversed())
                    .limit(limit)
                    .toList();
        } catch (RuntimeException exception) {
            log.warn("Cross-encoder reranking failed; falling back to RRF score order.", exception);
            return fallback.rerank(query, candidates, limit);
        }
    }
}
