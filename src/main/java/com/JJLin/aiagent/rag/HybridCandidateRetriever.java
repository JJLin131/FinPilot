package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Primary
@Component
public class HybridCandidateRetriever implements RagCandidateRetriever {

    private static final Logger log = LoggerFactory.getLogger(HybridCandidateRetriever.class);

    private final VectorCandidateRetriever vectorRetriever;
    private final Bm25CandidateRetriever bm25Retriever;
    private final int rrfK;

    public HybridCandidateRetriever(
            VectorCandidateRetriever vectorRetriever,
            Bm25CandidateRetriever bm25Retriever,
            KnowledgeProperties properties) {
        this.vectorRetriever = vectorRetriever;
        this.bm25Retriever = bm25Retriever;
        this.rrfK = Math.max(1, properties.getRrfK());
    }

    @Override
    public List<RagMatch> retrieve(KnowledgeDomain domain, String query, int limit) {
        Map<String, FusionEntry> fusion = new LinkedHashMap<>();
        addSafely(fusion, () -> vectorRetriever.retrieve(domain, query, limit), "vector");
        addSafely(fusion, () -> bm25Retriever.retrieve(domain, query, limit), "bm25");
        return fusion.values().stream()
                .map(entry -> new RagMatch(entry.match().documentId(), entry.match().title(), entry.match().source(),
                        entry.match().text(), entry.rrfScore()))
                .sorted(java.util.Comparator.comparingDouble(RagMatch::score).reversed())
                .limit(limit)
                .toList();
    }

    private void addSafely(Map<String, FusionEntry> fusion, RetrievalCall call, String source) {
        try {
            add(fusion, call.retrieve());
        } catch (RuntimeException exception) {
            log.warn("{} candidate retrieval failed; continuing with remaining retrievers.", source, exception);
        }
    }

    private void add(Map<String, FusionEntry> fusion, List<RagMatch> results) {
        for (int rank = 0; rank < results.size(); rank++) {
            RagMatch match = results.get(rank);
            String key = match.documentId() + ":" + match.text().hashCode();
            double score = 1.0 / (rrfK + rank + 1);
            fusion.merge(key, new FusionEntry(match, score),
                    (left, right) -> new FusionEntry(left.match(), left.rrfScore() + right.rrfScore()));
        }
    }

    private record FusionEntry(RagMatch match, double rrfScore) {
    }

    private interface RetrievalCall {
        List<RagMatch> retrieve();
    }
}
