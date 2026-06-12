package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.agent.RagCurationAssistant;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class RagCurationAgent {

    private static final Logger log = LoggerFactory.getLogger(RagCurationAgent.class);

    private final RagCurationAssistant assistant;

    public RagCurationAgent(RagCurationAssistant assistant) {
        this.assistant = assistant;
    }

    public KnowledgeDocumentRequest curate(KnowledgeDocumentRequest source) {
        CuratedKnowledgeDocument curated;
        try {
            curated = assistant.curate(source.getContent());
        } catch (RuntimeException exception) {
            log.warn("RAG document curation unavailable for {}; preserving the original document.",
                    source.getDocumentId(), exception);
            curated = null;
        }
        KnowledgeDocumentRequest result = new KnowledgeDocumentRequest();
        result.setDocumentId(source.getDocumentId());
        result.setDomain(source.getDomain());
        result.setTenantId(source.getTenantId());
        result.setSource(source.getSource());
        result.setValidFrom(source.getValidFrom());
        result.setValidTo(source.getValidTo());
        result.setTitle(nonBlank(curated == null ? null : curated.getTitle(), source.getTitle()));
        String normalizedMarkdown = nonBlank(curated == null ? null : curated.getNormalizedMarkdown(), source.getContent());
        result.setContent(preservesAllSourceParagraphs(source.getContent(), normalizedMarkdown)
                ? normalizedMarkdown
                : source.getContent());
        result.setTags(curated == null || curated.getTags() == null ? List.of() : curated.getTags());
        return result;
    }

    private String nonBlank(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private boolean preservesAllSourceParagraphs(String source, String curated) {
        if (source == null || curated == null) {
            return false;
        }
        return source.lines()
                .map(String::trim)
                .filter(line -> !line.isBlank())
                .allMatch(curated::contains);
    }
}
