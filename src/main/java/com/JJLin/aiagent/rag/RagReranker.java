package com.JJLin.aiagent.rag;

import java.util.List;

public interface RagReranker {
    List<RagMatch> rerank(String query, List<RagMatch> candidates, int limit);
}
