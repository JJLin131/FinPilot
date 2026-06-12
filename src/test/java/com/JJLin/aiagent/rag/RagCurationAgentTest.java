package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.agent.RagCurationAssistant;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RagCurationAgentTest {

    @Test
    void preservesOriginalContentAndTrustedScope() {
        RagCurationAssistant assistant = mock(RagCurationAssistant.class);
        CuratedKnowledgeDocument curated = new CuratedKnowledgeDocument();
        curated.setTitle("Structured policy");
        curated.setNormalizedMarkdown("# Policy\nOriginal clause");
        curated.setTags(List.of("refund", "policy"));
        when(assistant.curate("Original clause")).thenReturn(curated);
        KnowledgeDocumentRequest source = new KnowledgeDocumentRequest();
        source.setDocumentId("doc-1");
        source.setDomain(KnowledgeDomain.FINANCE);
        source.setTenantId("tenant-a");
        source.setTitle("Policy");
        source.setSource("contract.pdf");
        source.setContent("Original clause");

        KnowledgeDocumentRequest result = new RagCurationAgent(assistant).curate(source);

        assertEquals("# Policy\nOriginal clause", result.getContent());
        assertEquals("tenant-a", result.getTenantId());
        assertEquals(List.of("refund", "policy"), result.getTags());
    }

    @Test
    void fallsBackToOriginalWhenCurationDropsAClause() {
        RagCurationAssistant assistant = mock(RagCurationAssistant.class);
        CuratedKnowledgeDocument curated = new CuratedKnowledgeDocument();
        curated.setNormalizedMarkdown("Clause A");
        when(assistant.curate("Clause A\nClause B")).thenReturn(curated);
        KnowledgeDocumentRequest source = new KnowledgeDocumentRequest();
        source.setDocumentId("doc-2");
        source.setDomain(KnowledgeDomain.FINANCE);
        source.setTenantId("tenant-a");
        source.setTitle("Contract");
        source.setSource("contract.pdf");
        source.setContent("Clause A\nClause B");

        KnowledgeDocumentRequest result = new RagCurationAgent(assistant).curate(source);

        assertEquals("Clause A\nClause B", result.getContent());
    }

    @Test
    void fallsBackToOriginalWhenCurationServiceIsOffline() {
        RagCurationAssistant assistant = mock(RagCurationAssistant.class);
        when(assistant.curate("Original clause")).thenThrow(new IllegalStateException("deepseek offline"));
        KnowledgeDocumentRequest source = new KnowledgeDocumentRequest();
        source.setDocumentId("doc-3");
        source.setDomain(KnowledgeDomain.FINANCE);
        source.setTenantId("tenant-a");
        source.setTitle("Policy");
        source.setSource("policy.md");
        source.setContent("Original clause");

        KnowledgeDocumentRequest result = new RagCurationAgent(assistant).curate(source);

        assertEquals("Original clause", result.getContent());
        assertEquals("Policy", result.getTitle());
        assertEquals(List.of(), result.getTags());
    }
}
