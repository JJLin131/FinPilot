package com.JJLin.aiagent.rag;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class RagKnowledgeService {

    private static final Logger log = LoggerFactory.getLogger(RagKnowledgeService.class);

    private final EmbeddingModel embeddingModel;
    private final KnowledgeStoreRouter storeRouter;
    private final KnowledgeDocumentRegistry registry;
    private final KnowledgeChunker chunker;
    private final QueryRewriter queryRewriter;
    private final RagCandidateRetriever candidateRetriever;
    private final RagReranker reranker;
    private final Bm25ChunkIndex bm25Index;

    public RagKnowledgeService(
            EmbeddingModel knowledgeEmbeddingModel,
            KnowledgeStoreRouter storeRouter,
            KnowledgeDocumentRegistry registry,
            KnowledgeChunker chunker,
            QueryRewriter queryRewriter,
            RagCandidateRetriever candidateRetriever,
            RagReranker reranker,
            Bm25ChunkIndex bm25Index) {
        this.embeddingModel = knowledgeEmbeddingModel;
        this.storeRouter = storeRouter;
        this.registry = registry;
        this.chunker = chunker;
        this.queryRewriter = queryRewriter;
        this.candidateRetriever = candidateRetriever;
        this.reranker = reranker;
        this.bm25Index = bm25Index;
    }

    public KnowledgeDocumentResult ingest(KnowledgeDocumentRequest request) {
        validateTenant(request);
        List<String> oldChunkIds = registry.chunkIds(request.getDocumentId());
        if (!oldChunkIds.isEmpty()) {
            removeVectorChunksBestEffort(request.getDomain(), oldChunkIds);
        }
        List<String> chunks = chunker.split(request.getContent());
        List<String> chunkIds = new ArrayList<>();
        List<TextSegment> segments = new ArrayList<>();
        for (int index = 0; index < chunks.size(); index++) {
            String chunkId = deterministicChunkId(request, index);
            chunkIds.add(chunkId);
            Metadata metadata = new Metadata()
                    .put("documentId", request.getDocumentId())
                    .put("domain", request.getDomain().name())
                    .put("title", request.getTitle())
                    .put("source", request.getSource())
                    .put("chunkIndex", index);
            if (request.getTenantId() != null && !request.getTenantId().isBlank()) {
                metadata.put("tenantId", request.getTenantId());
            }
            if (request.getTags() != null && !request.getTags().isEmpty()) {
                metadata.put("tags", String.join(",", request.getTags()));
            }
            segments.add(TextSegment.from(chunks.get(index), metadata));
        }
        if (!segments.isEmpty()) {
            indexVectorsBestEffort(request, chunkIds, segments);
        }
        bm25Index.replaceDocument(request, chunkIds, chunks);
        registry.save(request, chunks.size());
        registry.replaceChunkIds(request.getDocumentId(), chunkIds);
        return new KnowledgeDocumentResult(request.getDocumentId(), request.getDomain(), request.getTenantId(),
                "ACTIVE", chunks.size(), request.getValidFrom(), request.getValidTo());
    }

    public List<RagMatch> search(KnowledgeDomain domain, String tenantId, String query, int limit) {
        Map<String, RagMatch> candidates = new LinkedHashMap<>();
        for (String rewrittenQuery : queryRewriter.rewrite(query)) {
            for (RagMatch candidate : candidateRetriever.retrieve(domain, rewrittenQuery, Math.max(limit * 4, 12))) {
                String documentId = candidate.documentId();
                if (!registry.isActive(documentId, domain, tenantId, LocalDate.now())) {
                    continue;
                }
                String key = documentId + ":" + candidate.text().hashCode();
                candidates.merge(key, candidate, (left, right) -> left.score() >= right.score() ? left : right);
            }
        }
        return reranker.rerank(query, new ArrayList<>(candidates.values()), limit);
    }

    public int deleteExpiredDocuments(LocalDate today) {
        List<String> expiredIds = registry.expiredDocumentIds(today);
        expiredIds.forEach(documentId -> {
            List<String> chunkIds = registry.chunkIds(documentId);
            if (!chunkIds.isEmpty()) {
                removeVectorChunksBestEffort(registry.domain(documentId), chunkIds);
            }
            bm25Index.deleteDocument(documentId);
            registry.markExpired(documentId);
        });
        return expiredIds.size();
    }

    private void indexVectorsBestEffort(
            KnowledgeDocumentRequest request,
            List<String> chunkIds,
            List<TextSegment> segments) {
        try {
            storeRouter.store(request.getDomain()).addAll(
                    chunkIds, embeddingModel.embedAll(segments).content(), segments);
        } catch (RuntimeException exception) {
            log.warn("Vector indexing unavailable for document {}; BM25 indexing will remain active.",
                    request.getDocumentId(), exception);
        }
    }

    private void removeVectorChunksBestEffort(KnowledgeDomain domain, List<String> chunkIds) {
        try {
            storeRouter.store(domain).removeAll(chunkIds);
        } catch (RuntimeException exception) {
            log.warn("Vector store unavailable while removing {} chunks; continuing with metadata and BM25 cleanup.",
                    chunkIds.size(), exception);
        }
    }

    private void validateTenant(KnowledgeDocumentRequest request) {
        if (request.getDomain() == KnowledgeDomain.FINANCE
                && (request.getTenantId() == null || request.getTenantId().isBlank())) {
            throw new IllegalArgumentException("Finance knowledge documents require tenantId.");
        }
    }

    private String deterministicChunkId(KnowledgeDocumentRequest request, int index) {
        String value = request.getDomain() + ":" + request.getTenantId() + ":" + request.getDocumentId() + ":" + index;
        return UUID.nameUUIDFromBytes(value.getBytes(StandardCharsets.UTF_8)).toString();
    }
}
