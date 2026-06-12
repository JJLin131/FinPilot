package com.JJLin.aiagent.rag;

import java.util.List;

public interface RagCandidateRetriever {
    List<RagMatch> retrieve(KnowledgeDomain domain, String query, int limit);
}
