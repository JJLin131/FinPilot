package com.JJLin.aiagent.config;

import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class KnowledgeConfigTest {

    @Test
    void configuresBgeM3ThroughOllama() {
        KnowledgeProperties properties = new KnowledgeProperties();

        var model = new KnowledgeConfig().knowledgeEmbeddingModel(properties);

        assertInstanceOf(OllamaEmbeddingModel.class, model);
    }
}
