package com.JJLin.aiagent.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.junit.jupiter.api.Test;

import java.time.LocalDate;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RagKnowledgeServiceTest {

    @Test
    void appliesQueryRewriteFiltersInactiveDocumentsAndReranks() {
        KnowledgeDocumentRegistry registry = mock(KnowledgeDocumentRegistry.class);
        QueryRewriter rewriter = query -> List.of(query, "rewritten");
        RagCandidateRetriever retriever = mock(RagCandidateRetriever.class);
        when(retriever.retrieve(eq(KnowledgeDomain.FINANCE), any(), any(Integer.class))).thenReturn(List.of(
                new RagMatch("active", "Active", "source-a", "supported", 0.7),
                new RagMatch("expired", "Expired", "source-b", "stale", 0.9)));
        when(registry.isActive(eq("active"), eq(KnowledgeDomain.FINANCE), eq("tenant-a"), any(LocalDate.class)))
                .thenReturn(true);
        when(registry.isActive(eq("expired"), eq(KnowledgeDomain.FINANCE), eq("tenant-a"), any(LocalDate.class)))
                .thenReturn(false);
        RagKnowledgeService service = service(registry, rewriter, retriever);

        var results = service.search(KnowledgeDomain.FINANCE, "tenant-a", "question", 5);

        assertEquals(List.of("active"), results.stream().map(RagMatch::documentId).toList());
        verify(retriever).retrieve(KnowledgeDomain.FINANCE, "question", 20);
        verify(retriever).retrieve(KnowledgeDomain.FINANCE, "rewritten", 20);
    }

    @Test
    void requiresTenantForFinanceDocuments() {
        RagKnowledgeService service = service(mock(KnowledgeDocumentRegistry.class),
                new IdentityQueryRewriter(), mock(RagCandidateRetriever.class));
        KnowledgeDocumentRequest request = new KnowledgeDocumentRequest();
        request.setDomain(KnowledgeDomain.FINANCE);

        assertThrows(IllegalArgumentException.class, () -> service.ingest(request));
    }

    @Test
    void keepsBm25AndMetadataWhenVectorIndexingIsOffline() {
        EmbeddingModel embeddingModel = mock(EmbeddingModel.class);
        when(embeddingModel.embedAll(any())).thenThrow(new IllegalStateException("ollama offline"));
        @SuppressWarnings("unchecked")
        EmbeddingStore<TextSegment> store = mock(EmbeddingStore.class);
        KnowledgeStoreRouter router = mock(KnowledgeStoreRouter.class);
        when(router.store(KnowledgeDomain.FINANCE)).thenReturn(store);
        KnowledgeDocumentRegistry registry = mock(KnowledgeDocumentRegistry.class);
        KnowledgeChunker chunker = mock(KnowledgeChunker.class);
        when(chunker.split("content")).thenReturn(List.of("chunk"));
        Bm25ChunkIndex bm25Index = mock(Bm25ChunkIndex.class);
        RagKnowledgeService service = new RagKnowledgeService(
                embeddingModel, router, registry, chunker, new IdentityQueryRewriter(),
                mock(RagCandidateRetriever.class), new ScoreRagReranker(), bm25Index);
        KnowledgeDocumentRequest request = new KnowledgeDocumentRequest();
        request.setDocumentId("doc-1");
        request.setDomain(KnowledgeDomain.FINANCE);
        request.setTenantId("tenant-a");
        request.setTitle("title");
        request.setSource("source");
        request.setContent("content");

        KnowledgeDocumentResult result = service.ingest(request);

        assertEquals("ACTIVE", result.status());
        verify(bm25Index).replaceDocument(eq(request), any(), eq(List.of("chunk")));
        verify(registry).save(request, 1);
        verify(registry).replaceChunkIds(eq("doc-1"), any());
    }

    private RagKnowledgeService service(
            KnowledgeDocumentRegistry registry,
            QueryRewriter rewriter,
            RagCandidateRetriever retriever) {
        return new RagKnowledgeService(
                mock(EmbeddingModel.class),
                mock(KnowledgeStoreRouter.class),
                registry,
                mock(KnowledgeChunker.class),
                rewriter,
                retriever,
                new ScoreRagReranker(),
                mock(Bm25ChunkIndex.class));
    }
}
