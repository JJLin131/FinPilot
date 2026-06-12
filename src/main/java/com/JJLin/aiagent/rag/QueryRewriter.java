package com.JJLin.aiagent.rag;

import java.util.List;

public interface QueryRewriter {
    List<String> rewrite(String query);
}
