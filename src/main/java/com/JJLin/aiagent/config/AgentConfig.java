package com.JJLin.aiagent.config;

import com.JJLin.aiagent.agent.FinanceAssistant;
import com.JJLin.aiagent.agent.IntentClassifierAssistant;
import com.JJLin.aiagent.agent.QueryAssistant;
import com.JJLin.aiagent.agent.RagCurationAssistant;
import com.JJLin.aiagent.agent.TransferAssistant;
import com.JJLin.aiagent.memory.JdbcAgentChatMemoryStore;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.AiServices;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AgentConfig {

    @Bean
    ChatMemoryProvider chatMemoryProvider(JdbcAgentChatMemoryStore store) {
        return memoryId -> MessageWindowChatMemory.builder()
                .id(memoryId)
                .chatMemoryStore(store)
                .maxMessages(20)
                .build();
    }

    @Bean
    RagCurationAssistant ragCurationAssistant(ChatModel chatModel) {
        return AiServices.builder(RagCurationAssistant.class)
                .chatModel(chatModel)
                .build();
    }

    @Bean
    IntentClassifierAssistant intentClassifierAssistant(ChatModel chatModel) {
        return AiServices.builder(IntentClassifierAssistant.class)
                .chatModel(chatModel)
                .build();
    }

    @Bean
    FinanceAssistant financeAssistant(
            ChatModel chatModel,
            ChatMemoryProvider chatMemoryProvider) {
        return AiServices.builder(FinanceAssistant.class)
                .chatModel(chatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .build();
    }

    @Bean
    TransferAssistant transferAssistant(
            ChatModel chatModel,
            ChatMemoryProvider chatMemoryProvider) {
        return AiServices.builder(TransferAssistant.class)
                .chatModel(chatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .build();
    }

    @Bean
    QueryAssistant queryAssistant(
            ChatModel chatModel,
            ChatMemoryProvider chatMemoryProvider) {
        return AiServices.builder(QueryAssistant.class)
                .chatModel(chatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .build();
    }
}
