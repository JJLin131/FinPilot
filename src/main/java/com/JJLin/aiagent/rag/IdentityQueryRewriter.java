package com.JJLin.aiagent.rag;

import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class IdentityQueryRewriter implements QueryRewriter {
    @Override
    public List<String> rewrite(String query) {
        return List.of(query);
    }
}
