package com.JJLin.aiagent.rag;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.Date;
import java.time.LocalDate;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

@Repository
public class KnowledgeDocumentRegistry {

    private final JdbcTemplate jdbcTemplate;

    public KnowledgeDocumentRegistry(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public void save(KnowledgeDocumentRequest request, int chunkCount) {
        jdbcTemplate.update("""
                insert into knowledge_document
                (document_id, domain, tenant_id, title, source_uri, content_hash, tags_text,
                 status, valid_from, valid_to, chunk_count, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, current_timestamp(6))
                on duplicate key update domain=values(domain), tenant_id=values(tenant_id), title=values(title),
                source_uri=values(source_uri), content_hash=values(content_hash), tags_text=values(tags_text), status='ACTIVE',
                valid_from=values(valid_from), valid_to=values(valid_to), chunk_count=values(chunk_count),
                updated_at=current_timestamp(6)
                """, request.getDocumentId(), request.getDomain().name(), request.getTenantId(), request.getTitle(),
                request.getSource(), contentHash(request.getContent()), tags(request),
                date(request.getValidFrom()), date(request.getValidTo()), chunkCount);
    }

    public void replaceChunkIds(String documentId, List<String> chunkIds) {
        jdbcTemplate.update("delete from knowledge_document_chunk where document_id = ?", documentId);
        chunkIds.forEach(chunkId -> jdbcTemplate.update(
                "insert into knowledge_document_chunk(document_id, chunk_id) values (?, ?)", documentId, chunkId));
    }

    public List<String> chunkIds(String documentId) {
        return jdbcTemplate.query("select chunk_id from knowledge_document_chunk where document_id = ?",
                (rs, rowNum) -> rs.getString(1), documentId);
    }

    public boolean isActive(String documentId, KnowledgeDomain domain, String tenantId, LocalDate today) {
        Integer count = jdbcTemplate.queryForObject("""
                select count(*) from knowledge_document
                where document_id = ? and domain = ? and status = 'ACTIVE'
                and (tenant_id = '__GLOBAL__' or tenant_id = ?)
                and (valid_from is null or valid_from <= ?)
                and (valid_to is null or valid_to >= ?)
                """, Integer.class, documentId, domain.name(), tenantId, Date.valueOf(today), Date.valueOf(today));
        return count != null && count > 0;
    }

    public List<String> expiredDocumentIds(LocalDate today) {
        return jdbcTemplate.query("""
                select document_id from knowledge_document
                where status = 'ACTIVE' and valid_to is not null and valid_to < ?
                """, (rs, rowNum) -> rs.getString(1), Date.valueOf(today));
    }

    public KnowledgeDomain domain(String documentId) {
        return KnowledgeDomain.valueOf(jdbcTemplate.queryForObject(
                "select domain from knowledge_document where document_id = ?", String.class, documentId));
    }

    public void markExpired(String documentId) {
        jdbcTemplate.update("update knowledge_document set status='EXPIRED', updated_at=current_timestamp(6) where document_id = ?",
                documentId);
    }

    private Date date(LocalDate value) {
        return value == null ? null : Date.valueOf(value);
    }

    private String tags(KnowledgeDocumentRequest request) {
        return request.getTags() == null ? "" : String.join(",", request.getTags());
    }

    private String contentHash(String content) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(content.getBytes(StandardCharsets.UTF_8));
            return java.util.HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is not available.", exception);
        }
    }
}
