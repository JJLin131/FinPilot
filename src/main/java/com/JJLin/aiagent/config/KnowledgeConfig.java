package com.JJLin.aiagent.config;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.http.client.spring.restclient.SpringRestClientBuilder;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaApiVersion;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Lazy;

import java.time.Duration;

@Configuration
@EnableConfigurationProperties(KnowledgeProperties.class)
public class KnowledgeConfig {

    @Bean
    EmbeddingModel knowledgeEmbeddingModel(KnowledgeProperties properties) {
        return OllamaEmbeddingModel.builder()
                .httpClientBuilder(new SpringRestClientBuilder())
                .baseUrl(properties.getEmbeddingBaseUrl())
                .modelName(properties.getEmbeddingModelName())
                .timeout(Duration.ofSeconds(60))
                .build();
    }

    @Bean
    @Lazy
    EmbeddingStore<TextSegment> financeKnowledgeStore(KnowledgeProperties properties) {
        return chromaStore(properties, properties.getFinanceCollection());
    }

    private EmbeddingStore<TextSegment> chromaStore(KnowledgeProperties properties, String collectionName) {
        return ChromaEmbeddingStore.builder()
                .apiVersion(ChromaApiVersion.V2)
                .httpClientBuilder(new SpringRestClientBuilder())
                .baseUrl(properties.getChromaBaseUrl())
                .collectionName(collectionName)
                .timeout(Duration.ofSeconds(5))
                .build();
    }
}
