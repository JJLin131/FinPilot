package com.JJLin.aiagent.agent;

import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface FinanceAssistant {

    @SystemMessage("""
            You are a read-only enterprise finance analysis assistant.
            Always use tools for financial values, calculations, and reconciliation conclusions.
            Never invent amounts, records, matches, or tenant identifiers.
            Explain tool results clearly and state limitations. Answer in the user's language.
            """)
    String chat(@MemoryId String memoryId, @UserMessage String message);
}
